from pathlib import Path
import pandas as pd


CANDIDATE_INPUT = Path(
    "outputs/16_methaneair_s2_candidate_events.csv"
)

PATCH_INDEX_INPUT = Path(
    "outputs/16_methaneair_s2_patch_index.csv"
)

AUDIT_OUTPUT = Path(
    "outputs/443_methaneair_s2_low_emission_candidate_audit_v1.csv"
)

REPORT_OUTPUT = Path(
    "outputs/444_methaneair_s2_low_emission_candidate_report_v1.txt"
)

LOW_EMISSION_LIMIT_KG_H = 1000.0


def main():
    candidates = pd.read_csv(
        CANDIDATE_INPUT,
        low_memory=False,
    )

    patches = pd.read_csv(
        PATCH_INDEX_INPUT,
        low_memory=False,
    )

    required_candidate_columns = [
        "event_id",
        "emission_kg_hr",
        "datetime_utc",
    ]

    missing = [
        column
        for column in required_candidate_columns
        if column not in candidates.columns
    ]

    if missing:
        raise KeyError(
            "Candidate table missing columns: "
            + ", ".join(missing)
        )

    candidates[
        "emission_kg_hr"
    ] = pd.to_numeric(
        candidates["emission_kg_hr"],
        errors="coerce",
    )

    low = candidates[
        candidates["emission_kg_hr"].gt(0)
        & candidates["emission_kg_hr"].lt(
            LOW_EMISSION_LIMIT_KG_H
        )
    ].copy()

    patch_columns = [
        column
        for column in [
            "event_id",
            "filename",
            "relative_path",
            "download_status",
            "datetime_utc",
            "emission_kg_hr",
        ]
        if column in patches.columns
    ]

    patch_lookup = patches[
        patch_columns
    ].copy()

    patch_lookup = patch_lookup.rename(
        columns={
            "filename":
                "downloaded_filename",

            "relative_path":
                "downloaded_patch_path",

            "download_status":
                "existing_download_status",

            "datetime_utc":
                "downloaded_datetime_utc",

            "emission_kg_hr":
                "downloaded_emission_kg_hr",
        }
    )

    # One row per downloaded event for the availability check.
    patch_lookup = patch_lookup.drop_duplicates(
        subset=["event_id"],
        keep="first",
    )

    audit = low.merge(
        patch_lookup,
        on="event_id",
        how="left",
        validate="many_to_one",
    )

    audit[
        "already_in_patch_index"
    ] = audit[
        "downloaded_filename"
    ].notna()

    if "downloaded_patch_path" in audit.columns:
        audit[
            "downloaded_patch_exists"
        ] = audit[
            "downloaded_patch_path"
        ].map(
            lambda value: (
                Path(str(value)).exists()
                if pd.notna(value)
                else False
            )
        )
    else:
        audit[
            "downloaded_patch_exists"
        ] = False

    audit[
        "low_emission_bin"
    ] = pd.cut(
        audit["emission_kg_hr"],
        bins=[
            0,
            200,
            500,
            1000,
        ],
        labels=[
            "0_to_200",
            "200_to_500",
            "500_to_1000",
        ],
        right=False,
    )

    audit = audit.sort_values(
        [
            "emission_kg_hr",
            "datetime_utc",
            "event_id",
        ]
    ).reset_index(drop=True)

    audit.to_csv(
        AUDIT_OUTPUT,
        index=False,
    )

    bin_summary = (
        audit.groupby(
            "low_emission_bin",
            observed=False,
        )
        .agg(
            candidate_rows=(
                "event_id",
                "size",
            ),
            unique_events=(
                "event_id",
                "nunique",
            ),
            already_in_patch_index=(
                "already_in_patch_index",
                "sum",
            ),
            local_patch_exists=(
                "downloaded_patch_exists",
                "sum",
            ),
            minimum_rate_kg_h=(
                "emission_kg_hr",
                "min",
            ),
            maximum_rate_kg_h=(
                "emission_kg_hr",
                "max",
            ),
        )
        .reset_index()
    )

    candidate_unique = int(
        audit["event_id"].nunique()
    )

    downloaded_unique = int(
        audit.loc[
            audit["already_in_patch_index"],
            "event_id",
        ].nunique()
    )

    existing_unique = int(
        audit.loc[
            audit["downloaded_patch_exists"],
            "event_id",
        ].nunique()
    )

    missing_unique = int(
        candidate_unique
        - downloaded_unique
    )

    report_lines = [
        "=" * 105,
        "METHANEAIR–S2 LOW-EMISSION CANDIDATE AUDIT V1",
        "=" * 105,
        "",
        (
            "Low-emission definition: "
            f"0 < emission < {LOW_EMISSION_LIMIT_KG_H} kg/h"
        ),
        (
            "Low-emission candidate rows: "
            f"{len(audit)}"
        ),
        (
            "Unique low-emission event IDs: "
            f"{candidate_unique}"
        ),
        (
            "Already represented in patch index: "
            f"{downloaded_unique}"
        ),
        (
            "Local patches currently existing: "
            f"{existing_unique}"
        ),
        (
            "Unique low-emission events missing "
            f"from patch index: {missing_unique}"
        ),
        "",
        "Emission-bin summary:",
        bin_summary.to_string(index=False),
        "",
        "Lowest-rate candidate events:",
        audit[
            [
                "event_id",
                "datetime_utc",
                "emission_kg_hr",
                "low_emission_bin",
                "already_in_patch_index",
                "downloaded_patch_exists",
            ]
        ]
        .head(20)
        .to_string(index=False),
    ]

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("=" * 105)
    print(
        "METHANEAIR–S2 LOW-EMISSION CANDIDATE AUDIT"
    )
    print("=" * 105)

    print(
        "\nLow-emission candidate rows:",
        len(audit),
    )

    print(
        "Unique low-emission events:",
        candidate_unique,
    )

    print(
        "Already in patch index:",
        downloaded_unique,
    )

    print(
        "Local patches existing:",
        existing_unique,
    )

    print(
        "Missing from patch index:",
        missing_unique,
    )

    print("\nEmission-bin summary:")
    print(
        bin_summary.to_string(index=False)
    )

    print("\nSaved:")
    print(AUDIT_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
