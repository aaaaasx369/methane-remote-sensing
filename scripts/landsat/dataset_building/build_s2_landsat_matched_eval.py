#!/usr/bin/env python
"""Build fair S2-only, Landsat-only, and S2+Landsat MethaneFuse eval tables.

The script:
1. Reads row-level S2 and Landsat prediction tables.
2. Restricts both sensors to physical controlled-release labels.
3. Creates one-to-one, same-site, same-label temporal pairs.
4. Uses exact shared canonical keys first when available.
5. Otherwise pairs acquisitions on the same UTC date by minimum time difference.
6. Writes identical-query eval CSVs for S2-only, L8/9-only, and native
   MethaneFuse learned fusion.
7. Also writes a strict QA80 subset where both sensors pass local quality checks.

This script deliberately does NOT pair observations merely because their labels
or emission bins match.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


S2_PATH_COLUMNS = ["s2_0_path", "s2_90_path", "s2_360_path"]
L89_PATH_COLUMNS = ["l89_0_path", "l89_90_path", "l89_360_path"]

CANONICAL_KEY_CANDIDATES = [
    "release_interval_id",
    "canonical_event_id",
    "controlled_release_event_id",
    "source_event_id",
    "observation_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s2_predictions", required=True, type=Path)
    parser.add_argument("--l89_predictions", required=True, type=Path)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--max_hours",
        type=float,
        default=24.0,
        help="Maximum S2-Landsat acquisition-time difference.",
    )
    parser.add_argument(
        "--allow_cross_date",
        action="store_true",
        help=(
            "Allow fallback temporal pairs across different UTC dates. "
            "By default, fallback pairs must be on the same UTC date."
        ),
    )
    parser.add_argument(
        "--min_s2_clear",
        type=float,
        default=0.80,
        help="Strict-cohort minimum S2 local SCL clear fraction.",
    )
    parser.add_argument(
        "--min_l89_clear",
        type=float,
        default=0.80,
        help="Strict-cohort minimum Landsat local QA clear fraction.",
    )
    return parser.parse_args()


def first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def parse_time(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, utc=True, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        return series.map(lambda value: pd.to_datetime(value, utc=True, errors="coerce"))


def parse_bool(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def nonempty(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    return str(value).strip().lower() not in {"", "nan", "none", "null"}


def normalize_site(value: Any) -> str:
    text = str(value).strip().lower()
    compact = re.sub(r"[^a-z0-9]", "", text)
    if "casagrande" in compact:
        return "Casa_Grande"
    if "ehrenberg" in compact:
        return "Ehrenberg"
    if "methaneair038" in compact or compact.endswith("038"):
        return "MA_site_038"
    if "methaneair043" in compact or compact.endswith("043"):
        return "MA_site_043"
    if "methaneair073" in compact or compact.endswith("073"):
        return "MA_site_073"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")


def standardize_sensor_table(df: pd.DataFrame, sensor: str) -> pd.DataFrame:
    out = df.copy()

    label_col = first_existing(out, ["true_label", "label", "model_label", "physical_release_gt"])
    site_col = first_existing(out, ["site", "model_site", "site_normalized"])
    id_col = first_existing(
        out,
        ["external_eval_id", "prediction_sample_id", "sample_id", "id", "model_scene_key", "scene_id"],
    )

    if sensor == "s2":
        time_col = first_existing(out, ["acquisition_time_utc", "model_acquisition_time_utc"])
        required_paths = S2_PATH_COLUMNS
    else:
        time_col = first_existing(out, ["model_acquisition_time_utc", "acquisition_time_utc"])
        required_paths = L89_PATH_COLUMNS

    missing = {
        "label": label_col,
        "site": site_col,
        "time": time_col,
        "id": id_col,
    }
    unresolved = [name for name, column in missing.items() if column is None]
    if unresolved:
        raise ValueError(
            f"{sensor} table cannot resolve required fields {unresolved}. "
            f"Columns={out.columns.tolist()}"
        )

    missing_paths = [column for column in required_paths if column not in out.columns]
    if missing_paths:
        raise ValueError(
            f"{sensor} table is missing MethaneFuse path columns {missing_paths}. "
            "Use the row-level prediction CSV produced from the corresponding eval manifest."
        )

    assert label_col and site_col and time_col and id_col

    out[f"{sensor}_row_id"] = out[id_col].astype(str)
    out["match_label"] = pd.to_numeric(out[label_col], errors="coerce")
    out["match_site"] = out[site_col].map(normalize_site)
    out[f"{sensor}_time"] = parse_time(out[time_col])

    # Only real controlled-release binary labels belong in the fair benchmark.
    provenance_col = first_existing(out, ["label_provenance", "ground_truth_type"])
    if provenance_col is not None:
        provenance = out[provenance_col].astype(str).str.strip().str.lower()
        keep = provenance.eq("physical_release") | provenance.str.contains(
            "controlled_release", na=False
        )
        out = out[keep].copy()

    out = out.dropna(subset=["match_label", "match_site", f"{sensor}_time"]).copy()
    out["match_label"] = out["match_label"].astype(int)
    out = out[out["match_label"].isin([0, 1])].copy()

    # Require all temporal frames to be available.
    ready = pd.Series(True, index=out.index)
    for column in required_paths:
        ready &= out[column].map(nonempty)
    out = out[ready].copy()

    # Quality-pass flag retained for strict matched subset.
    if sensor == "s2":
        if "scl80_pass" in out.columns:
            out["sensor_quality_pass"] = parse_bool(out["scl80_pass"])
        elif "minimum_scl_clear_fraction" in out.columns:
            out["sensor_quality_pass"] = (
                pd.to_numeric(out["minimum_scl_clear_fraction"], errors="coerce") >= 0.80
            )
        else:
            out["sensor_quality_pass"] = False
    else:
        if "qa80_pass" in out.columns:
            out["sensor_quality_pass"] = parse_bool(out["qa80_pass"])
        elif "minimum_qa_clear_fraction" in out.columns:
            out["sensor_quality_pass"] = (
                pd.to_numeric(out["minimum_qa_clear_fraction"], errors="coerce") >= 0.80
            )
        else:
            out["sensor_quality_pass"] = False

    return out.reset_index(drop=True)


def overlapping_canonical_keys(s2: pd.DataFrame, l89: pd.DataFrame) -> list[str]:
    keys: list[str] = []
    for column in CANONICAL_KEY_CANDIDATES:
        if column not in s2.columns or column not in l89.columns:
            continue
        s_values = {str(v).strip() for v in s2[column] if nonempty(v)}
        l_values = {str(v).strip() for v in l89[column] if nonempty(v)}
        if s_values & l_values:
            keys.append(column)
    return keys


def candidate_pairs(
    s2: pd.DataFrame,
    l89: pd.DataFrame,
    max_hours: float,
    allow_cross_date: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    exact_keys = overlapping_canonical_keys(s2, l89)

    for s_idx, s_row in s2.iterrows():
        subset = l89[
            l89["match_site"].eq(s_row["match_site"])
            & l89["match_label"].eq(s_row["match_label"])
        ]

        for l_idx, l_row in subset.iterrows():
            delta_hours = abs(
                (s_row["s2_time"] - l_row["l89_time"]).total_seconds()
            ) / 3600.0

            exact_key = None
            for key in exact_keys:
                if nonempty(s_row.get(key)) and nonempty(l_row.get(key)):
                    if str(s_row.get(key)).strip() == str(l_row.get(key)).strip():
                        exact_key = key
                        break

            same_date = s_row["s2_time"].date() == l_row["l89_time"].date()
            temporal_ok = delta_hours <= max_hours and (allow_cross_date or same_date)

            if exact_key is None and not temporal_ok:
                continue

            if exact_key is not None:
                method = f"exact_{exact_key}"
                priority = 0
            elif delta_hours <= 6:
                method = "same_utc_date_within_6h"
                priority = 1
            else:
                method = "same_utc_date_within_24h" if same_date else "near_time_cross_date"
                priority = 2

            rows.append(
                {
                    "s2_index": int(s_idx),
                    "l89_index": int(l_idx),
                    "match_site": s_row["match_site"],
                    "label": int(s_row["match_label"]),
                    "s2_time": s_row["s2_time"],
                    "l89_time": l_row["l89_time"],
                    "time_difference_hours": float(delta_hours),
                    "same_utc_date": bool(same_date),
                    "match_method": method,
                    "match_priority": priority,
                }
            )

    return pd.DataFrame(rows)


def greedy_one_to_one(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()

    ordered = candidates.sort_values(
        ["match_priority", "time_difference_hours", "match_site", "s2_time", "l89_time"]
    )

    used_s2: set[int] = set()
    used_l89: set[int] = set()
    selected: list[pd.Series] = []

    for _, row in ordered.iterrows():
        s_idx = int(row["s2_index"])
        l_idx = int(row["l89_index"])
        if s_idx in used_s2 or l_idx in used_l89:
            continue
        used_s2.add(s_idx)
        used_l89.add(l_idx)
        selected.append(row)

    if not selected:
        return pd.DataFrame(columns=candidates.columns)
    return pd.DataFrame(selected).reset_index(drop=True)


def prefixed_metadata(row: pd.Series, prefix: str, excluded: set[str]) -> dict[str, Any]:
    return {
        f"{prefix}_{column}": value
        for column, value in row.items()
        if column not in excluded
    }


def build_pair_table(
    s2: pd.DataFrame,
    l89: pd.DataFrame,
    selected: pd.DataFrame,
    min_s2_clear: float,
    min_l89_clear: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for pair_number, pair in enumerate(selected.itertuples(index=False), start=1):
        s_row = s2.loc[int(pair.s2_index)]
        l_row = l89.loc[int(pair.l89_index)]

        s2_clear = pd.to_numeric(
            pd.Series([s_row.get("minimum_scl_clear_fraction")]), errors="coerce"
        ).iloc[0]
        l89_clear = pd.to_numeric(
            pd.Series([l_row.get("minimum_qa_clear_fraction")]), errors="coerce"
        ).iloc[0]

        s2_quality = bool(s_row.get("sensor_quality_pass", False))
        l89_quality = bool(l_row.get("sensor_quality_pass", False))

        if np.isfinite(s2_clear):
            s2_quality = bool(s2_clear >= min_s2_clear)
        if np.isfinite(l89_clear):
            l89_quality = bool(l89_clear >= min_l89_clear)

        base = {
            "pair_id": f"s2_l89_pair_{pair_number:03d}",
            "label": int(pair.label),
            "site": pair.match_site,
            "label_provenance": "physical_release",
            "match_method": pair.match_method,
            "time_difference_hours": float(pair.time_difference_hours),
            "same_utc_date": bool(pair.same_utc_date),
            "s2_acquisition_time_utc": pair.s2_time,
            "l89_acquisition_time_utc": pair.l89_time,
            "s2_row_id": s_row["s2_row_id"],
            "l89_row_id": l_row["l89_row_id"],
            "s2_quality_pass": s2_quality,
            "l89_quality_pass": l89_quality,
            "both_quality_pass": bool(s2_quality and l89_quality),
        }

        for column in S2_PATH_COLUMNS:
            base[column] = s_row[column]
        for column in L89_PATH_COLUMNS:
            base[column] = l_row[column]

        # Keep useful sensor-specific audit metadata without collisions.
        keep_s2 = [
            "probability_positive",
            "predicted_label",
            "scene_id",
            "event_id",
            "release_interval_id",
            "emission_rate_kg_hr_raw",
            "positive_emission_rate_kg_hr",
            "background_class",
            "minimum_scl_clear_fraction",
        ]
        keep_l89 = [
            "probability_positive",
            "predicted_label",
            "model_scene_key",
            "model_sensor",
            "release_rate_kg_h",
            "ch4_kgh_mean",
            "minimum_qa_clear_fraction",
        ]
        for column in keep_s2:
            if column in s_row.index:
                base[f"s2_{column}"] = s_row[column]
        for column in keep_l89:
            if column in l_row.index:
                base[f"l89_{column}"] = l_row[column]

        rows.append(base)

    return pd.DataFrame(rows)


def eval_columns(pair_table: pd.DataFrame, mode: str) -> pd.DataFrame:
    base_columns = [
        "pair_id",
        "label",
        "label_provenance",
        "site",
        "match_method",
        "time_difference_hours",
        "same_utc_date",
        "s2_acquisition_time_utc",
        "l89_acquisition_time_utc",
        "s2_row_id",
        "l89_row_id",
        "s2_quality_pass",
        "l89_quality_pass",
        "both_quality_pass",
    ]

    if mode == "s2":
        sensor_columns = S2_PATH_COLUMNS
    elif mode == "l89":
        sensor_columns = L89_PATH_COLUMNS
    elif mode == "fusion":
        sensor_columns = S2_PATH_COLUMNS + L89_PATH_COLUMNS
    else:
        raise ValueError(mode)

    extra_columns = [
        column
        for column in pair_table.columns
        if column.startswith("s2_") or column.startswith("l89_")
    ]

    columns = []
    for column in base_columns + sensor_columns + extra_columns:
        if column in pair_table.columns and column not in columns:
            columns.append(column)

    out = pair_table[columns].copy()
    out = out.rename(columns={"pair_id": "id"})
    out.insert(1, "sample_id", out["id"])
    return out


def write_cohort(
    pair_table: pd.DataFrame,
    output_dir: Path,
    cohort: str,
) -> list[Path]:
    paths: list[Path] = []
    for mode in ["s2", "l89", "fusion"]:
        path = output_dir / f"610_{cohort}_matched_{mode}_eval.csv"
        eval_columns(pair_table, mode).to_csv(path, index=False)
        paths.append(path)
    return paths


def class_counts(df: pd.DataFrame) -> str:
    if df.empty:
        return "NONE"
    return df["label"].value_counts(dropna=False).sort_index().to_string()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    s2_raw = pd.read_csv(args.s2_predictions, low_memory=False)
    l89_raw = pd.read_csv(args.l89_predictions, low_memory=False)

    s2 = standardize_sensor_table(s2_raw, "s2")
    l89 = standardize_sensor_table(l89_raw, "l89")

    candidates = candidate_pairs(
        s2=s2,
        l89=l89,
        max_hours=float(args.max_hours),
        allow_cross_date=bool(args.allow_cross_date),
    )
    selected = greedy_one_to_one(candidates)
    pair_table = build_pair_table(
        s2=s2,
        l89=l89,
        selected=selected,
        min_s2_clear=float(args.min_s2_clear),
        min_l89_clear=float(args.min_l89_clear),
    )

    raw_pairs_path = args.output_dir / "610_s2_landsat_matched_pairs_raw.csv"
    strict_pairs_path = args.output_dir / "611_s2_landsat_matched_pairs_qa80.csv"
    candidate_path = args.output_dir / "609_s2_landsat_pair_candidates.csv"
    report_path = args.output_dir / "612_s2_landsat_matching_report.txt"

    candidates.to_csv(candidate_path, index=False)
    pair_table.to_csv(raw_pairs_path, index=False)

    strict_pairs = pair_table[pair_table["both_quality_pass"]].copy()
    strict_pairs.to_csv(strict_pairs_path, index=False)

    created = [candidate_path, raw_pairs_path, strict_pairs_path]
    created.extend(write_cohort(pair_table, args.output_dir, "raw"))
    created.extend(write_cohort(strict_pairs, args.output_dir, "qa80"))

    report_lines = [
        "Fair Sentinel-2 / Landsat matched benchmark preparation",
        "=" * 72,
        f"S2 input: {args.s2_predictions}",
        f"Landsat input: {args.l89_predictions}",
        f"S2 physical-release rows ready: {len(s2)}",
        f"Landsat physical-release rows ready: {len(l89)}",
        f"Candidate temporal/key pairs: {len(candidates)}",
        f"Selected one-to-one raw pairs: {len(pair_table)}",
        f"Selected one-to-one QA80 pairs: {len(strict_pairs)}",
        "",
        "Raw matched labels:",
        class_counts(pair_table),
        "",
        "QA80 matched labels:",
        class_counts(strict_pairs),
        "",
        "Match methods:",
        (
            pair_table["match_method"].value_counts(dropna=False).to_string()
            if len(pair_table)
            else "NONE"
        ),
        "",
        "Important interpretation:",
        "- All three model conditions use identical pair IDs and labels.",
        "- Native fusion eval rows contain both s2_* and l89_* temporal paths.",
        "- Exact canonical-key matches are preferred when shared keys exist.",
        "- Fallback pairs are same-site, same-label, one-to-one temporal matches.",
        "- By default fallback matches must occur on the same UTC date.",
        "- A small or empty cohort is a scientific result: current data do not",
        "  support a strong paired cross-sensor comparison.",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    created.append(report_path)

    print("\n".join(report_lines))
    print("\nCreated:")
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
