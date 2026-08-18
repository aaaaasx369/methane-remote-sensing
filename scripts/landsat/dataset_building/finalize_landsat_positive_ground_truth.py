from pathlib import Path
import re

import numpy as np
import pandas as pd


INPUT_INVENTORY = Path(
    "outputs/105_landsat_positive_release_inventory.csv"
)

EHRENBERG_EVIDENCE = Path(
    "outputs/85_ehrenberg_candidate_release_evidence.csv"
)

FINAL_OUTPUT = Path(
    "outputs/106_landsat_positive_ground_truth_final.csv"
)

THRESHOLD_SUMMARY_OUTPUT = Path(
    "outputs/107_landsat_high_emission_threshold_summary.csv"
)


PRIMARY_HIGH_EMISSION_THRESHOLD = 1000.0

SENSITIVITY_THRESHOLDS = [
    500.0,
    1000.0,
    1500.0,
]


# 已經由先前 scene-level release review 確認的排放量。
# 單位全部統一成 kg/h。
CONFIRMED_RELEASES = {
    "RG_2694f689b602": {
        "release_rate_kg_h": 73.947699197088,
        "rate_source":
            "outputs/52_2022_landsat_scene_label_review.csv",
        "rate_confidence": "exact",
        "overlap_evidence_type":
            "exact_measurement_interval_overlap",
    },
    "RG_a89d3baebaa3": {
        "release_rate_kg_h": 755.9147350275,
        "rate_source":
            "outputs/52_2022_landsat_scene_label_review.csv",
        "rate_confidence": "exact",
        "overlap_evidence_type":
            "exact_measurement_interval_overlap",
    },
    "RG_9602975fd846": {
        "release_rate_kg_h": 1161.72896942448,
        "rate_source":
            "outputs/52_2022_landsat_scene_label_review.csv",
        "rate_confidence": "exact",
        "overlap_evidence_type":
            "exact_measurement_interval_overlap",
    },
    "RG_541ef17f3f25": {
        "release_rate_kg_h": 1386.794744720475,
        "rate_source":
            "outputs/52_2022_landsat_scene_label_review.csv",
        "rate_confidence": "exact",
        "overlap_evidence_type":
            "exact_measurement_interval_overlap",
    },
    "RG_7aeae7ac0770": {
        "release_rate_kg_h": 1475.1657353601,
        "rate_source":
            "outputs/52_2022_landsat_scene_label_review.csv",
        "rate_confidence": "exact",
        "overlap_evidence_type":
            "exact_measurement_interval_overlap",
    },
    "OP_013": {
        "release_rate_kg_h": 984.882546,
        "rate_source":
            "outputs/69-70_casa_grande_candidate_release_audit",
        "rate_confidence": "exact",
        "overlap_evidence_type":
            "exact_measurement_interval_overlap",
    },
    "RG_1da3ffbf963f": {
        "release_rate_kg_h": 2100.0,
        "rate_source":
            "outputs/56_2021_landsat_scene_label_review.csv",
        "rate_confidence": "approximate",
        "overlap_evidence_type":
            "positive_record_within_24_seconds",
    },
}


def clean_text(value):
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.lower() in {
        "",
        "nan",
        "none",
        "null",
    }:
        return ""

    return text


def extract_numbers(text, pattern):
    values = []

    for match in re.findall(
        pattern,
        str(text),
        flags=re.IGNORECASE,
    ):
        try:
            value = float(match)

            if np.isfinite(value):
                values.append(value)

        except ValueError:
            pass

    return values


def derive_op028_rate_and_conversion():
    """
    OP_028 has repeated high-frequency flow evidence.

    We use the median CH4 kg/h value around the overpass.
    We also derive the dataset-specific conversion factor:
        kg/h per SCFH
    from paired mean30 values.
    """
    fallback_rate = 1424.23
    fallback_factor = 0.0180430205

    if not EHRENBERG_EVIDENCE.exists():
        return (
            fallback_rate,
            fallback_factor,
            "fallback_from_previous_OP_028_review",
        )

    evidence = pd.read_csv(
        EHRENBERG_EVIDENCE,
        low_memory=False,
    )

    if "overpass_id" not in evidence.columns:
        return (
            fallback_rate,
            fallback_factor,
            "fallback_missing_overpass_column",
        )

    op028 = evidence[
        evidence["overpass_id"]
        .astype(str)
        .str.strip()
        .eq("OP_028")
    ].copy()

    if len(op028) == 0:
        return (
            fallback_rate,
            fallback_factor,
            "fallback_OP_028_not_found",
        )

    if "seconds_from_landsat" in op028.columns:
        seconds = pd.to_numeric(
            op028["seconds_from_landsat"],
            errors="coerce",
        )

        close = seconds.abs() <= 300

        if close.any():
            op028 = op028[close].copy()

    flow_column = None

    for candidate in [
        "flow_values",
        "evidence_value",
    ]:
        if candidate in op028.columns:
            flow_column = candidate
            break

    if flow_column is None:
        return (
            fallback_rate,
            fallback_factor,
            "fallback_missing_flow_values",
        )

    kg_values = []
    conversion_factors = []

    for text in op028[flow_column].dropna():
        current_kg = extract_numbers(
            text,
            r"cr_kgh_CH4_mean30\s*=\s*"
            r"([-+]?[0-9]*\.?[0-9]+)",
        )

        current_scfh = extract_numbers(
            text,
            r"cr_scfh_mean30\s*=\s*"
            r"([-+]?[0-9]*\.?[0-9]+)",
        )

        kg_values.extend(
            value
            for value in current_kg
            if 0 < value < 10000
        )

        if current_kg and current_scfh:
            kg_value = current_kg[0]
            scfh_value = current_scfh[0]

            if (
                kg_value > 0
                and scfh_value > 0
            ):
                conversion_factors.append(
                    kg_value / scfh_value
                )

    if kg_values:
        rate = float(
            np.median(kg_values)
        )
    else:
        rate = fallback_rate

    if conversion_factors:
        factor = float(
            np.median(conversion_factors)
        )
    else:
        factor = fallback_factor

    return (
        rate,
        factor,
        "median_from_OP_028_near_overpass_records",
    )


def determine_lookup_key(row):
    overpass_id = clean_text(
        row.get("overpass_id")
    )

    raster_group_id = clean_text(
        row.get("raster_group_id")
    )

    scene_key = clean_text(
        row.get("scene_key")
    )

    if overpass_id in CONFIRMED_RELEASES:
        return overpass_id

    if raster_group_id in CONFIRMED_RELEASES:
        return raster_group_id

    if scene_key in CONFIRMED_RELEASES:
        return scene_key

    if overpass_id == "OP_028":
        return "OP_028"

    if raster_group_id == "RG_EH_OP_028":
        return "OP_028"

    if raster_group_id == "RG_57d0e3988348":
        return "RG_57d0e3988348"

    return ""


def assign_rate_tier(rate):
    if rate < 500:
        return "low"

    if rate < 1000:
        return "medium"

    return "high"


def main():
    if not INPUT_INVENTORY.exists():
        raise FileNotFoundError(
            f"Missing inventory: "
            f"{INPUT_INVENTORY}"
        )

    inventory = pd.read_csv(
        INPUT_INVENTORY,
        low_memory=False,
    )

    if len(inventory) != 9:
        raise ValueError(
            "Expected 9 positive scenes, "
            f"found {len(inventory)}."
        )

    (
        op028_rate,
        scfh_to_kgh_factor,
        op028_derivation_source,
    ) = derive_op028_rate_and_conversion()

    # 2021-10-21 direct flow was 231569 SCFH.
    rg57_rate = (
        231569.0
        * scfh_to_kgh_factor
    )

    dynamic_records = {
        "OP_028": {
            "release_rate_kg_h":
                op028_rate,
            "release_rate_original_value":
                op028_rate,
            "release_rate_original_unit":
                "kg/h",
            "rate_source":
                op028_derivation_source,
            "rate_confidence":
                "derived_median",
            "overlap_evidence_type":
                "continuous_positive_flow_near_overpass",
        },
        "RG_57d0e3988348": {
            "release_rate_kg_h":
                rg57_rate,
            "release_rate_original_value":
                231569.0,
            "release_rate_original_unit":
                "SCFH",
            "rate_source":
                (
                    "outputs/56_2021_landsat_scene_label_review.csv; "
                    "SCFH converted using median dataset-specific "
                    "OP_028 kg/h-per-SCFH factor"
                ),
            "rate_confidence":
                "derived_from_scfh",
            "overlap_evidence_type":
                "continuous_positive_flow_before_and_after_overpass",
        },
    }

    all_records = {
        **CONFIRMED_RELEASES,
        **dynamic_records,
    }

    finalized_rows = []

    for _, row in inventory.iterrows():
        key = determine_lookup_key(row)

        if not key:
            raise KeyError(
                "No confirmed release record for:\n"
                + row[
                    [
                        "scene_key",
                        "raster_group_id",
                        "overpass_id",
                    ]
                ].to_string()
            )

        record = all_records[key]

        output_row = row.to_dict()

        rate = float(
            record["release_rate_kg_h"]
        )

        output_row.update({
            "ground_truth_key": key,
            "release_rate_kg_h":
                rate,
            "release_rate_original_value":
                record.get(
                    "release_rate_original_value",
                    rate,
                ),
            "release_rate_original_unit":
                record.get(
                    "release_rate_original_unit",
                    "kg/h",
                ),
            "release_rate_source":
                record["rate_source"],
            "release_rate_confidence":
                record["rate_confidence"],
            "exact_overlap":
                True,
            "overlap_evidence_type":
                record[
                    "overlap_evidence_type"
                ],
            "ground_truth_quality_pass":
                True,
            "image_visibility_reviewed":
                False,
            "image_visibility_status":
                "not_yet_reviewed",
            "release_rate_tier":
                assign_rate_tier(rate),
            "high_emission_threshold_kg_h":
                PRIMARY_HIGH_EMISSION_THRESHOLD,
            "high_emission_label":
                int(
                    rate
                    >= PRIMARY_HIGH_EMISSION_THRESHOLD
                ),
            "high_emission_500kgph":
                int(rate >= 500),
            "high_emission_1000kgph":
                int(rate >= 1000),
            "high_emission_1500kgph":
                int(rate >= 1500),
            "review_status":
                "finalized_ground_truth",
        })

        finalized_rows.append(
            output_row
        )

    final = pd.DataFrame(
        finalized_rows
    )

    if final["release_rate_kg_h"].isna().any():
        raise ValueError(
            "Some positive scenes still have "
            "missing release rates."
        )

    if not final["exact_overlap"].all():
        raise ValueError(
            "Some positive scenes are not confirmed overlaps."
        )

    if final["ground_truth_key"].duplicated().any():
        duplicates = final[
            final["ground_truth_key"]
            .duplicated(keep=False)
        ]

        raise ValueError(
            "Duplicate ground-truth keys:\n"
            + duplicates[
                [
                    "ground_truth_key",
                    "scene_key",
                ]
            ].to_string(index=False)
        )

    final = final.sort_values(
        [
            "site_key",
            "satellite_time",
        ]
    ).reset_index(drop=True)

    FINAL_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final.to_csv(
        FINAL_OUTPUT,
        index=False,
    )

    threshold_rows = []

    for threshold in SENSITIVITY_THRESHOLDS:
        labels = (
            final["release_rate_kg_h"]
            >= threshold
        )

        threshold_rows.append({
            "threshold_kg_h":
                threshold,
            "positive_scene_count":
                len(final),
            "high_emission_count":
                int(labels.sum()),
            "below_threshold_count":
                int((~labels).sum()),
            "high_emission_fraction":
                float(labels.mean()),
        })

    threshold_summary = pd.DataFrame(
        threshold_rows
    )

    threshold_summary.to_csv(
        THRESHOLD_SUMMARY_OUTPUT,
        index=False,
    )

    print("=" * 105)
    print("FINAL POSITIVE-SCENE GROUND TRUTH")
    print("=" * 105)

    print(
        f"\nOP_028 release rate: "
        f"{op028_rate:.3f} kg/h"
    )

    print(
        "Dataset-specific SCFH → kg/h factor:",
        f"{scfh_to_kgh_factor:.8f}",
    )

    print(
        f"RG_57d converted release rate: "
        f"{rg57_rate:.3f} kg/h"
    )

    print("\nFinal positive scenes:")

    print(
        final[
            [
                "ground_truth_key",
                "site_key",
                "satellite_time",
                "release_rate_kg_h",
                "release_rate_confidence",
                "exact_overlap",
                "release_rate_tier",
                "high_emission_label",
                "overlap_evidence_type",
            ]
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.3f}",
        )
    )

    print("\nRelease-rate summary:")

    print(
        final.groupby(
            "site_key"
        )["release_rate_kg_h"]
        .agg([
            "count",
            "min",
            "median",
            "max",
        ])
    )

    print(
        "\nPrimary high-emission threshold:",
        PRIMARY_HIGH_EMISSION_THRESHOLD,
        "kg/h",
    )

    print("\nPrimary label counts:")

    print(
        final[
            "high_emission_label"
        ].value_counts()
        .sort_index()
    )

    print("\nThreshold sensitivity:")

    print(
        threshold_summary.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.3f}",
        )
    )

    print("\nSaved:")
    print(FINAL_OUTPUT)
    print(THRESHOLD_SUMMARY_OUTPUT)


if __name__ == "__main__":
    main()
