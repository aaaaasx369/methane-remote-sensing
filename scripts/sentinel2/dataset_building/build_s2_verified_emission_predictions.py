from __future__ import annotations

from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


PROJECT = Path("/Users/happydoraaa/methane_release_project")
OUTPUTS = PROJECT / "outputs"

PREDICTIONS_PATH = OUTPUTS / "77_s2_loso_oof_predictions.csv"
MODEL_SUMMARY_PATH = OUTPUTS / "78_s2_loso_summary.csv"
INTERVALS_PATH = OUTPUTS / "309_all_exact_release_intervals_for_s2.csv"

ALL_AUDIT_PATH = OUTPUTS / "80_s2_release_overlap_audit_all.csv"
VERIFIED_PATH = OUTPUTS / "80_s2_verified_emission_predictions.csv"
AMBIGUOUS_PATH = OUTPUTS / "80_s2_ambiguous_release_matches.csv"
SUMMARY_PATH = OUTPUTS / "80_s2_verified_emission_predictions_summary.txt"

MODEL_OVERRIDE: str | None = None


def normalize_site(value: object) -> str:
    text = str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def choose_model(predictions: pd.DataFrame) -> str:
    available = predictions["model"].dropna().astype(str).unique().tolist()

    if MODEL_OVERRIDE is not None:
        if MODEL_OVERRIDE not in available:
            raise ValueError(
                f"MODEL_OVERRIDE={MODEL_OVERRIDE!r} 不在 77 檔案中。\n"
                f"Available models: {available}"
            )
        return MODEL_OVERRIDE

    if MODEL_SUMMARY_PATH.exists():
        summary = pd.read_csv(MODEL_SUMMARY_PATH, low_memory=False)

        if {"model", "balanced_accuracy"}.issubset(summary.columns):
            candidates = summary[
                ~summary["model"].astype(str).str.contains(
                    "dummy", case=False, na=False
                )
            ].copy()

            candidates["balanced_accuracy"] = pd.to_numeric(
                candidates["balanced_accuracy"], errors="coerce"
            )

            if "roc_auc" in candidates.columns:
                candidates["roc_auc"] = pd.to_numeric(
                    candidates["roc_auc"], errors="coerce"
                )
            else:
                candidates["roc_auc"] = np.nan

            candidates = candidates[
                candidates["model"].astype(str).isin(available)
            ].sort_values(
                ["balanced_accuracy", "roc_auc"],
                ascending=False,
                na_position="last",
            )

            if len(candidates):
                return str(candidates.iloc[0]["model"])

    non_dummy = [
        model for model in available
        if "dummy" not in model.lower()
    ]

    if non_dummy:
        return non_dummy[0]

    if available:
        return available[0]

    raise ValueError("77 檔案中沒有可用的 model。")


def require_columns(
    df: pd.DataFrame,
    required: list[str],
    file_label: str,
) -> None:
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            f"{file_label} 缺少欄位：{missing}\n"
            f"目前欄位：{df.columns.tolist()}"
        )


def main() -> None:
    for path in [PREDICTIONS_PATH, INTERVALS_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"找不到檔案：{path}")

    predictions = pd.read_csv(PREDICTIONS_PATH, low_memory=False)
    intervals = pd.read_csv(INTERVALS_PATH, low_memory=False)

    require_columns(
        predictions,
        [
            "model",
            "held_out_site",
            "sample_id",
            "scene_id",
            "acquisition_time_utc",
            "true_label",
            "predicted_label",
            "positive_probability",
        ],
        "77_s2_loso_oof_predictions.csv",
    )

    require_columns(
        intervals,
        [
            "release_start_utc",
            "release_end_utc",
            "release_rate_kg_h",
            "site",
            "release_interval_id",
        ],
        "309_all_exact_release_intervals_for_s2.csv",
    )

    selected_model = choose_model(predictions)
    predictions = predictions[
        predictions["model"].astype(str) == selected_model
    ].copy()

    predictions["acquisition_time_utc"] = pd.to_datetime(
        predictions["acquisition_time_utc"],
        utc=True,
        errors="coerce",
    )

    intervals["release_start_utc"] = pd.to_datetime(
        intervals["release_start_utc"],
        utc=True,
        errors="coerce",
    )
    intervals["release_end_utc"] = pd.to_datetime(
        intervals["release_end_utc"],
        utc=True,
        errors="coerce",
    )
    intervals["release_rate_kg_h"] = pd.to_numeric(
        intervals["release_rate_kg_h"],
        errors="coerce",
    )

    predictions["_site_key"] = predictions["held_out_site"].map(
        normalize_site
    )
    intervals["_site_key"] = intervals["site"].map(normalize_site)

    bad_predictions = predictions[
        predictions["acquisition_time_utc"].isna()
    ]

    if len(bad_predictions):
        raise ValueError(
            "77 中有無法解析的 acquisition_time_utc：\n"
            + bad_predictions[
                ["sample_id", "acquisition_time_utc"]
            ].to_string(index=False)
        )

    intervals = intervals.dropna(
        subset=[
            "release_start_utc",
            "release_end_utc",
            "release_rate_kg_h",
        ]
    ).copy()

    invalid_intervals = intervals[
        intervals["release_end_utc"]
        <= intervals["release_start_utc"]
    ]

    if len(invalid_intervals):
        raise ValueError(
            "309 中存在 end <= start 的 interval：\n"
            + invalid_intervals[
                [
                    "release_interval_id",
                    "release_start_utc",
                    "release_end_utc",
                ]
            ].head(20).to_string(index=False)
        )

    interval_identity = [
        "_site_key",
        "release_start_utc",
        "release_end_utc",
        "release_rate_kg_h",
    ]

    sort_columns = []
    ascending = []

    if "rate_priority" in intervals.columns:
        sort_columns.append("rate_priority")
        ascending.append(True)

    if "release_duration_minutes" in intervals.columns:
        sort_columns.append("release_duration_minutes")
        ascending.append(True)

    if sort_columns:
        intervals = intervals.sort_values(
            sort_columns,
            ascending=ascending,
            na_position="last",
        )

    intervals = intervals.drop_duplicates(
        subset=interval_identity,
        keep="first",
    )

    audit_rows: list[dict] = []
    ambiguous_rows: list[dict] = []

    for _, prediction in predictions.iterrows():
        acquisition_time = prediction["acquisition_time_utc"]
        site_key = prediction["_site_key"]

        site_intervals = intervals[
            intervals["_site_key"] == site_key
        ]

        matches = site_intervals[
            (site_intervals["release_start_utc"] <= acquisition_time)
            & (acquisition_time < site_intervals["release_end_utc"])
        ].copy()

        base = prediction.drop(labels=["_site_key"]).to_dict()
        base["selected_model"] = selected_model
        base["original_dataset_label"] = int(
            prediction["true_label"]
        )

        if len(matches) == 0:
            audit_rows.append({
                **base,
                "match_status": "no_exact_release_interval",
                "exact_release_overlap": False,
                "matched_interval_count": 0,
                "release_interval_id": pd.NA,
                "release_start_utc": pd.NaT,
                "release_end_utc": pd.NaT,
                "metered_release_rate_kg_hr": np.nan,
                "physical_release_gt": pd.NA,
                "true_label": pd.NA,
                "label_agrees_with_physical_gt": pd.NA,
            })
            continue

        unique_rates = (
            matches["release_rate_kg_h"]
            .round(9)
            .dropna()
            .unique()
        )

        if len(matches) > 1 and len(unique_rates) > 1:
            for _, match in matches.iterrows():
                ambiguous_rows.append({
                    **base,
                    "match_status": "ambiguous_multiple_intervals",
                    "release_interval_id":
                        match["release_interval_id"],
                    "release_start_utc":
                        match["release_start_utc"],
                    "release_end_utc":
                        match["release_end_utc"],
                    "metered_release_rate_kg_hr":
                        match["release_rate_kg_h"],
                    "release_rate_source":
                        match.get("release_rate_source"),
                    "source_file": match.get("source_file"),
                })

            audit_rows.append({
                **base,
                "match_status": "ambiguous_multiple_intervals",
                "exact_release_overlap": False,
                "matched_interval_count": len(matches),
                "release_interval_id": pd.NA,
                "release_start_utc": pd.NaT,
                "release_end_utc": pd.NaT,
                "metered_release_rate_kg_hr": np.nan,
                "physical_release_gt": pd.NA,
                "true_label": pd.NA,
                "label_agrees_with_physical_gt": pd.NA,
            })
            continue

        match = matches.iloc[0]

        release_rate = float(match["release_rate_kg_h"])
        physical_gt = int(release_rate > 0)

        match_status = (
            "exact_unique_interval"
            if len(matches) == 1
            else "exact_equivalent_duplicate_intervals"
        )

        audit_rows.append({
            **base,
            "match_status": match_status,
            "exact_release_overlap": True,
            "matched_interval_count": len(matches),
            "release_interval_id":
                match["release_interval_id"],
            "release_start_utc":
                match["release_start_utc"],
            "release_end_utc":
                match["release_end_utc"],
            "release_duration_minutes":
                match.get("release_duration_minutes"),
            "metered_release_rate_kg_hr": release_rate,
            "release_rate_source":
                match.get("release_rate_source"),
            "release_rate_priority":
                match.get("rate_priority"),
            "release_interval_emission_bin":
                match.get("emission_bin"),
            "release_interval_strict_candidate":
                match.get("strict_interval_candidate"),
            "release_ground_truth_site":
                match.get("site"),
            "release_source_file":
                match.get("source_file"),
            "release_source_sheet":
                match.get("source_sheet"),
            "physical_release_gt": physical_gt,
            "true_label": physical_gt,
            "label_agrees_with_physical_gt": (
                int(prediction["true_label"]) == physical_gt
            ),
        })

    audit = pd.DataFrame(audit_rows)
    verified = audit[
        audit["exact_release_overlap"] == True
    ].copy()

    ambiguous = pd.DataFrame(ambiguous_rows)

    preferred_order = [
        "sample_id",
        "scene_id",
        "held_out_site",
        "acquisition_time_utc",
        "selected_model",
        "predicted_label",
        "positive_probability",
        "true_label",
        "physical_release_gt",
        "original_dataset_label",
        "label_agrees_with_physical_gt",
        "metered_release_rate_kg_hr",
        "release_start_utc",
        "release_end_utc",
        "release_interval_id",
        "exact_release_overlap",
        "match_status",
        "image_path",
    ]

    def reorder(df: pd.DataFrame) -> pd.DataFrame:
        first = [column for column in preferred_order if column in df.columns]
        rest = [column for column in df.columns if column not in first]
        return df[first + rest]

    audit = reorder(audit)
    verified = reorder(verified)

    audit.to_csv(ALL_AUDIT_PATH, index=False)
    verified.to_csv(VERIFIED_PATH, index=False)

    if len(ambiguous):
        ambiguous.to_csv(AMBIGUOUS_PATH, index=False)
    else:
        pd.DataFrame(
            columns=[
                "sample_id",
                "match_status",
                "release_interval_id",
                "release_start_utc",
                "release_end_utc",
                "metered_release_rate_kg_hr",
            ]
        ).to_csv(AMBIGUOUS_PATH, index=False)

    summary_lines = [
        "S2 exact controlled-release overlap audit",
        "=" * 70,
        f"Selected model: {selected_model}",
        f"Prediction rows after model filtering: {len(predictions)}",
        f"Release intervals after cleaning: {len(intervals)}",
        f"Exact verified S2 overlaps: {len(verified)}",
        "",
        "Match status:",
        audit["match_status"].value_counts(dropna=False).to_string(),
        "",
        "Verified rows by site:",
        (
            verified["held_out_site"].value_counts(dropna=False).to_string()
            if len(verified)
            else "NONE"
        ),
        "",
        "Verified physical ground truth:",
        (
            verified["physical_release_gt"]
            .value_counts(dropna=False)
            .sort_index()
            .to_string()
            if len(verified)
            else "NONE"
        ),
        "",
        "Verified emission-rate summary (kg/h):",
        (
            verified["metered_release_rate_kg_hr"]
            .describe()
            .to_string()
            if len(verified)
            else "NONE"
        ),
        "",
        "Important:",
        "Only rows in 80_s2_verified_emission_predictions.csv are eligible",
        "for exact release-rate analysis. Rows with no exact interval are not",
        "confirmed physical negatives and must not be assigned 0 kg/h.",
    ]

    SUMMARY_PATH.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print("\n" + "\n".join(summary_lines))
    print("\nCreated:")
    print(ALL_AUDIT_PATH)
    print(VERIFIED_PATH)
    print(AMBIGUOUS_PATH)
    print(SUMMARY_PATH)

    if len(verified):
        print("\nVerified rows:")
        display_columns = [
            "sample_id",
            "held_out_site",
            "acquisition_time_utc",
            "metered_release_rate_kg_hr",
            "physical_release_gt",
            "predicted_label",
            "positive_probability",
            "match_status",
        ]
        print(
            verified[display_columns]
            .sort_values("acquisition_time_utc")
            .to_string(index=False)
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"\nERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise
