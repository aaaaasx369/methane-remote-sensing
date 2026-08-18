from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "outputs/200_carbonmapper_plume_catalog_raw.csv"
)

CANDIDATE_OUTPUT = Path(
    "outputs/206_carbonmapper_ch4_ge1000_timed_candidates.csv"
)

SUMMARY_OUTPUT = Path(
    "outputs/207_carbonmapper_ch4_ge1000_candidate_summary.csv"
)

REJECT_OUTPUT = Path(
    "outputs/208_carbonmapper_ch4_ge1000_exclusion_audit.csv"
)

HIGH_EMISSION_THRESHOLD_KG_H = 1000.0


def nonempty(series):
    return (
        series.notna()
        & series.astype(str).str.strip().ne("")
        & ~series.astype(str).str.lower().eq("nan")
    )


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    df["gas_normalized"] = (
        df["gas"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    status_series = df.get(
        "status",
        pd.Series(
            "missing",
            index=df.index,
            dtype=str,
        ),
    )

    df["status_normalized"] = (
        status_series
        .fillna("missing")
        .astype(str)
        .str.strip()
        .str.lower()
        .replace("", "missing")
    )

    df["emission_auto"] = pd.to_numeric(
        df["emission_auto"],
        errors="coerce",
    )

    df["emission_uncertainty_auto"] = pd.to_numeric(
        df["emission_uncertainty_auto"],
        errors="coerce",
    )

    df["plume_latitude"] = pd.to_numeric(
        df["plume_latitude"],
        errors="coerce",
    )

    df["plume_longitude"] = pd.to_numeric(
        df["plume_longitude"],
        errors="coerce",
    )

    df["scene_datetime_utc"] = pd.to_datetime(
        df["scene_timestamp"],
        errors="coerce",
        utc=True,
    )

    df["published_datetime_utc"] = pd.to_datetime(
        df["published_at"],
        errors="coerce",
        utc=True,
    )

    product_columns = [
        column
        for column in [
            "plume_tif",
            "con_tif",
            "plume_png",
            "plume_rgb_png",
            "rgb_png",
        ]
        if column in df.columns
    ]

    df["has_plume_product"] = (
        df[product_columns]
        .apply(nonempty)
        .any(axis=1)
    )

    df["is_ch4"] = (
        df["gas_normalized"] == "CH4"
    )

    df["emission_ge_1000"] = (
        df["emission_auto"]
        >= HIGH_EMISSION_THRESHOLD_KG_H
    )

    df["has_coordinates"] = (
        df["plume_latitude"].between(
            -90,
            90,
        )
        & df["plume_longitude"].between(
            -180,
            180,
        )
    )

    df["has_scene_time"] = (
        df["scene_datetime_utc"].notna()
    )

    df["relative_emission_uncertainty"] = (
        df["emission_uncertainty_auto"]
        / df["emission_auto"]
    )

    base_high_emission = (
        df["is_ch4"]
        & df["emission_ge_1000"]
    )

    eligible = (
        base_high_emission
        & df["has_coordinates"]
        & df["has_scene_time"]
        & df["has_plume_product"]
    )

    df["candidate_eligible"] = eligible

    reasons = []

    for _, row in df.iterrows():
        row_reasons = []

        if not row["is_ch4"]:
            row_reasons.append("not_ch4")

        if not row["emission_ge_1000"]:
            row_reasons.append(
                "below_1000_kg_h"
            )

        if not row["has_coordinates"]:
            row_reasons.append(
                "missing_coordinates"
            )

        if not row["has_scene_time"]:
            row_reasons.append(
                "missing_scene_timestamp"
            )

        if not row["has_plume_product"]:
            row_reasons.append(
                "missing_plume_product"
            )

        reasons.append(
            "|".join(row_reasons)
            if row_reasons
            else "eligible"
        )

    df["exclusion_reason"] = reasons

    candidates = df[
        df["candidate_eligible"]
    ].copy()

    candidates["candidate_label"] = 1

    candidates["ground_truth_type"] = (
        "carbon_mapper_reported_ch4_plume"
    )

    candidates["metadata_quality_class"] = (
        "quality_field_unavailable"
    )

    candidates["manual_review_status"] = (
        "not_reviewed"
    )

    candidates["landsat_match_status"] = (
        "not_searched"
    )

    candidates = (
        candidates.sort_values(
            [
                "emission_auto",
                "scene_datetime_utc",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .drop_duplicates(
            subset=[
                "plume_id",
                "scene_id",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    exclusions = df[
        base_high_emission
        & ~df["candidate_eligible"]
    ].copy()

    summary = (
        candidates.groupby(
            [
                "status_normalized",
                "instrument",
                "platform",
            ],
            dropna=False,
        )
        .agg(
            candidate_count=(
                "plume_id",
                "size",
            ),
            median_emission_kg_h=(
                "emission_auto",
                "median",
            ),
            maximum_emission_kg_h=(
                "emission_auto",
                "max",
            ),
            median_relative_uncertainty=(
                "relative_emission_uncertainty",
                "median",
            ),
            first_scene=(
                "scene_datetime_utc",
                "min",
            ),
            last_scene=(
                "scene_datetime_utc",
                "max",
            ),
        )
        .reset_index()
        .sort_values(
            "candidate_count",
            ascending=False,
        )
    )

    CANDIDATE_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates.to_csv(
        CANDIDATE_OUTPUT,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    exclusions.to_csv(
        REJECT_OUTPUT,
        index=False,
    )

    print("=" * 105)
    print("CARBON MAPPER CANDIDATES WITHOUT QUALITY FIELD")
    print("=" * 105)

    print(
        "\nAll metadata rows:",
        len(df),
    )

    print(
        "CH4 >= 1000 kg/h:",
        int(base_high_emission.sum()),
    )

    print(
        "Eligible timed candidates:",
        len(candidates),
    )

    print(
        "Excluded high-emission candidates:",
        len(exclusions),
    )

    print("\nCandidate status values:")
    print(
        candidates[
            "status_normalized"
        ].value_counts(
            dropna=False
        )
    )

    print("\nCandidates by instrument:")
    print(
        candidates["instrument"]
        .value_counts(
            dropna=False
        )
        .head(20)
    )

    print("\nScene date range:")
    print(
        candidates[
            "scene_datetime_utc"
        ].min()
    )
    print(
        candidates[
            "scene_datetime_utc"
        ].max()
    )

    print("\nEmission summary:")
    print(
        candidates[
            "emission_auto"
        ].describe()
    )

    print("\nExclusion reasons:")
    print(
        exclusions[
            "exclusion_reason"
        ].value_counts(
            dropna=False
        )
    )

    print("\nSaved:")
    print(CANDIDATE_OUTPUT)
    print(SUMMARY_OUTPUT)
    print(REJECT_OUTPUT)


if __name__ == "__main__":
    main()
