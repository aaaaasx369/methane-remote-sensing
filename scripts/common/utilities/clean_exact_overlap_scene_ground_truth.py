from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


PROJECT = Path("/Users/happydoraaa/methane_release_project")
INPUT = PROJECT / "outputs/83_all_exact_satellite_release_overlaps.csv"

SCENE_OUTPUT = PROJECT / "outputs/85_exact_satellite_scene_ground_truth_clean.csv"
AMBIGUOUS_OUTPUT = PROJECT / "outputs/85_exact_satellite_scene_ground_truth_ambiguous.csv"
OVERPASS_AUDIT_OUTPUT = PROJECT / "outputs/86_possible_duplicate_overpass_audit.csv"
SUMMARY_OUTPUT = PROJECT / "outputs/86_scene_ground_truth_cleaning_summary.txt"

# Differences no larger than 1% are treated as equivalent numerical/source
# representations, not meaningfully different physical release rates.
RELATIVE_RATE_TOLERANCE = 0.01


def to_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def choose_representative(group: pd.DataFrame) -> pd.Series:
    work = group.copy()

    # Lower rate_priority is preferred when available.
    if "rate_priority" not in work.columns:
        work["rate_priority"] = np.nan

    # Preserve exact operation-log candidates when possible, but do not
    # discard zero-rate intervals merely because strict_interval_candidate=False.
    if "strict_interval_candidate" in work.columns:
        work["_strict_sort"] = work["strict_interval_candidate"].map(to_bool)
    else:
        work["_strict_sort"] = False

    work = work.sort_values(
        ["rate_priority", "_strict_sort", "release_duration_minutes"],
        ascending=[True, False, False],
        na_position="last",
    )

    return work.iloc[0]


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"找不到輸入檔：{INPUT}")

    df = pd.read_csv(INPUT, low_memory=False)

    required = [
        "sensor",
        "scene_id",
        "site",
        "acquisition_time_utc",
        "release_interval_id",
        "release_rate_kg_h",
        "physical_release_gt",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要欄位：{missing}")

    df["release_rate_kg_h"] = pd.to_numeric(
        df["release_rate_kg_h"], errors="coerce"
    )
    df["physical_release_gt"] = pd.to_numeric(
        df["physical_release_gt"], errors="coerce"
    )
    df["acquisition_time_utc"] = pd.to_datetime(
        df["acquisition_time_utc"], utc=True, errors="coerce"
    )

    scene_rows = []

    for (sensor, scene_id), group in df.groupby(
        ["sensor", "scene_id"], dropna=False, sort=True
    ):
        rates = (
            group["release_rate_kg_h"]
            .dropna()
            .astype(float)
            .to_numpy()
        )
        labels = (
            group["physical_release_gt"]
            .dropna()
            .astype(int)
            .unique()
        )

        if len(rates) == 0:
            rate_min = rate_max = rate_median = np.nan
            relative_range = np.nan
        else:
            rate_min = float(np.min(rates))
            rate_max = float(np.max(rates))
            rate_median = float(np.median(rates))
            denominator = max(abs(rate_median), 1e-12)
            relative_range = float((rate_max - rate_min) / denominator)

        if len(labels) > 1:
            status = "binary_ground_truth_conflict"
            binary_usable = False
            rate_usable = False
        elif len(group) == 1:
            status = "unique_scene_interval"
            binary_usable = True
            rate_usable = True
        elif np.isfinite(relative_range) and relative_range <= 1e-12:
            status = "multiple_intervals_same_rate"
            binary_usable = True
            rate_usable = True
        elif (
            np.isfinite(relative_range)
            and relative_range <= RELATIVE_RATE_TOLERANCE
        ):
            status = "equivalent_rates_within_1pct"
            binary_usable = True
            rate_usable = True
        else:
            status = "ambiguous_rates_over_1pct"
            binary_usable = len(labels) == 1
            rate_usable = False

        representative = choose_representative(group).copy()

        # For equivalent source/numerical duplicates, use the median as a
        # transparent consensus rate while retaining the selected source row.
        if status in {
            "multiple_intervals_same_rate",
            "equivalent_rates_within_1pct",
        }:
            consensus_rate = rate_median
        elif status == "unique_scene_interval":
            consensus_rate = float(representative["release_rate_kg_h"])
        else:
            consensus_rate = np.nan

        row = representative.to_dict()
        row.update(
            {
                "scene_rows_before_cleaning": int(len(group)),
                "unique_release_intervals": int(
                    group["release_interval_id"].nunique(dropna=True)
                ),
                "unique_release_rates": int(
                    group["release_rate_kg_h"].nunique(dropna=True)
                ),
                "release_interval_ids_all": " | ".join(
                    sorted(
                        set(
                            group["release_interval_id"]
                            .dropna()
                            .astype(str)
                        )
                    )
                ),
                "release_rates_all_kg_h": " | ".join(
                    f"{value:.6f}" for value in sorted(set(rates.tolist()))
                ),
                "release_rate_min_kg_h": rate_min,
                "release_rate_median_kg_h": rate_median,
                "release_rate_max_kg_h": rate_max,
                "relative_rate_range": relative_range,
                "scene_ground_truth_status": status,
                "binary_gt_usable": binary_usable,
                "emission_rate_usable": rate_usable,
                "consensus_release_rate_kg_h": consensus_rate,
            }
        )
        scene_rows.append(row)

    scene = pd.DataFrame(scene_rows)

    preferred = [
        "sensor",
        "scene_id",
        "site",
        "acquisition_time_utc",
        "physical_release_gt",
        "binary_gt_usable",
        "emission_rate_usable",
        "consensus_release_rate_kg_h",
        "scene_ground_truth_status",
        "scene_rows_before_cleaning",
        "unique_release_intervals",
        "unique_release_rates",
        "release_interval_ids_all",
        "release_rates_all_kg_h",
        "release_rate_min_kg_h",
        "release_rate_median_kg_h",
        "release_rate_max_kg_h",
        "relative_rate_range",
    ]
    scene = scene[
        [c for c in preferred if c in scene.columns]
        + [c for c in scene.columns if c not in preferred]
    ].sort_values(["sensor", "site", "acquisition_time_utc"])

    scene.to_csv(SCENE_OUTPUT, index=False)

    ambiguous = scene[
        ~scene["emission_rate_usable"]
    ].copy()
    ambiguous.to_csv(AMBIGUOUS_OUTPUT, index=False)

    # Audit likely duplicate representations of one physical overpass.
    # This does not automatically delete them; it only identifies groups
    # within the same sensor, site, release interval and 10-minute window.
    overpass = scene.copy()
    overpass["_time_10min"] = (
        overpass["acquisition_time_utc"].dt.floor("10min")
    )
    overpass_groups = (
        overpass.groupby(
            [
                "sensor",
                "site",
                "release_interval_id",
                "_time_10min",
            ],
            dropna=False,
        )
        .agg(
            scene_count=("scene_id", "nunique"),
            scene_ids=(
                "scene_id",
                lambda x: " | ".join(sorted(set(map(str, x)))),
            ),
            acquisition_times=(
                "acquisition_time_utc",
                lambda x: " | ".join(
                    sorted(set(x.astype(str)))
                ),
            ),
            physical_release_gt=("physical_release_gt", "first"),
            consensus_release_rate_kg_h=(
                "consensus_release_rate_kg_h",
                "median",
            ),
        )
        .reset_index()
    )
    overpass_groups = overpass_groups[
        overpass_groups["scene_count"] > 1
    ].copy()
    overpass_groups.to_csv(OVERPASS_AUDIT_OUTPUT, index=False)

    status_table = (
        scene["scene_ground_truth_status"]
        .value_counts(dropna=False)
        .to_string()
    )

    sensor_table = (
        scene.groupby("sensor")
        .agg(
            unique_scenes=("scene_id", "nunique"),
            binary_gt_usable=("binary_gt_usable", "sum"),
            emission_rate_usable=("emission_rate_usable", "sum"),
        )
        .reset_index()
        .to_string(index=False)
    )

    summary = [
        "Scene-level exact-overlap ground-truth cleaning",
        "=" * 72,
        f"Input scene-interval rows: {len(df)}",
        f"Unique scene-level records: {len(scene)}",
        f"Binary GT usable scenes: {int(scene['binary_gt_usable'].sum())}",
        f"Emission-rate usable scenes: {int(scene['emission_rate_usable'].sum())}",
        f"Emission-rate ambiguous scenes: {len(ambiguous)}",
        "",
        "Scene status:",
        status_table,
        "",
        "By sensor:",
        sensor_table,
        "",
        f"Possible duplicate-overpass groups: {len(overpass_groups)}",
        "",
        "Interpretation:",
        "- Binary ON/OFF evaluation may use rows with binary_gt_usable=True.",
        "- Exact emission-rate evaluation may use only emission_rate_usable=True.",
        "- Possible duplicate overpasses must be reviewed before claiming",
        "  independent acquisition counts.",
    ]

    SUMMARY_OUTPUT.write_text("\n".join(summary), encoding="utf-8")

    print("\n".join(summary))
    print("\nCreated:")
    print(SCENE_OUTPUT)
    print(AMBIGUOUS_OUTPUT)
    print(OVERPASS_AUDIT_OUTPUT)
    print(SUMMARY_OUTPUT)


if __name__ == "__main__":
    main()
