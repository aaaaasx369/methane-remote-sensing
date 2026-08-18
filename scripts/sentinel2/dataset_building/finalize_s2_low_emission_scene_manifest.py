from pathlib import Path
import numpy as np
import pandas as pd


DETAIL_INPUT = Path(
    "outputs/314_s2_exact_low_emission_provenance.csv"
)

OUTPUT = Path(
    "outputs/317_s2_low_emission_scene_manifest_v1.csv"
)

PRIMARY_OUTPUT = Path(
    "outputs/318_s2_low_emission_primary_scenes_v1.csv"
)


def emission_bin(rate):
    if pd.isna(rate):
        return "missing"
    if rate <= 0:
        return "zero"
    if rate < 200:
        return "0_to_200"
    if rate < 500:
        return "200_to_500"
    if rate < 1000:
        return "500_to_1000"
    if rate < 2000:
        return "1000_to_2000"
    return "2000_plus"


def require_one(frame, message):
    if len(frame) != 1:
        raise RuntimeError(
            f"{message}: expected one row, found {len(frame)}"
        )
    return frame.iloc[0]


def make_record(
    row,
    *,
    review_status,
    primary_include,
    final_rate,
    selected_interval,
    excluded_intervals="",
    conflict=False,
    notes="",
):
    return {
        "site":
            row["site"],

        "scene_id":
            row["best_scene_id"],

        "acquisition_time_utc":
            row["best_acquisition_time_utc"],

        "lat":
            row.get("lat"),

        "lon":
            row.get("lon"),

        "selected_release_interval_id":
            selected_interval,

        "excluded_release_interval_ids":
            excluded_intervals,

        "release_start_utc":
            row["release_start_utc"],

        "release_end_utc":
            row["release_end_utc"],

        "final_release_rate_kg_h":
            float(final_rate),

        "final_emission_bin":
            emission_bin(final_rate),

        "rate_source":
            row.get("release_rate_source"),

        "source_file":
            row.get("source_file"),

        "source_sheet":
            row.get("source_sheet"),

        "scene_cloud_percentage":
            row.get(
                "best_cloudy_pixel_percentage"
            ),

        "physical_release_label":
            1,

        "benchmark_label":
            1 if primary_include else np.nan,

        "primary_include":
            primary_include,

        "ground_truth_conflict":
            conflict,

        "local_qa_status":
            "pending",

        "review_status":
            review_status,

        "review_notes":
            notes,

        "manifest_version":
            "s2_low_emission_scene_v1",
    }


def main():
    detail = pd.read_csv(
        DETAIL_INPUT,
        low_memory=False,
    )

    detail[
        "best_acquisition_time_utc"
    ] = pd.to_datetime(
        detail[
            "best_acquisition_time_utc"
        ],
        errors="coerce",
        utc=True,
    )

    detail[
        "release_start_utc"
    ] = pd.to_datetime(
        detail[
            "release_start_utc"
        ],
        errors="coerce",
        utc=True,
    )

    detail[
        "release_end_utc"
    ] = pd.to_datetime(
        detail[
            "release_end_utc"
        ],
        errors="coerce",
        utc=True,
    )

    detail[
        "release_rate_kg_h"
    ] = pd.to_numeric(
        detail[
            "release_rate_kg_h"
        ],
        errors="coerce",
    )

    detail["acquisition_date"] = (
        detail[
            "best_acquisition_time_utc"
        ].dt.strftime("%Y-%m-%d")
    )

    records = []

    # ---------------------------------------------------------
    # 2021-08-03:
    # keep current file; discard superseded version.
    # ---------------------------------------------------------
    august = detail[
        detail["acquisition_date"].eq(
            "2021-08-03"
        )
    ].copy()

    superseded_flag = (
        august["source_file"]
        .fillna("")
        .astype(str)
        .str.contains(
            r"superceded|superseded",
            case=False,
            regex=True,
        )
    )

    current_august = august[
        ~superseded_flag
    ]

    old_august = august[
        superseded_flag
    ]

    august_row = require_one(
        current_august,
        "Current 2021-08-03 source",
    )

    excluded_august_ids = " | ".join(
        old_august[
            "release_interval_id"
        ]
        .dropna()
        .astype(str)
        .tolist()
    )

    records.append(
        make_record(
            august_row,
            review_status=
                "strict_positive_current_source",
            primary_include=True,
            final_rate=
                august_row[
                    "release_rate_kg_h"
                ],
            selected_interval=
                august_row[
                    "release_interval_id"
                ],
            excluded_intervals=
                excluded_august_ids,
            conflict=False,
            notes=(
                "Kept the current non-superseded "
                "controlled-release record. The "
                "75.97 kg/h record was excluded "
                "because its file is stored under "
                "the superseded directory. Rates "
                "were not summed."
            ),
        )
    )

    # ---------------------------------------------------------
    # 2021-11-03:
    # clean single strict low-emission scene.
    # ---------------------------------------------------------
    november_2021 = detail[
        detail["acquisition_date"].eq(
            "2021-11-03"
        )
    ]

    november_2021_row = require_one(
        november_2021,
        "2021-11-03 scene",
    )

    records.append(
        make_record(
            november_2021_row,
            review_status=
                "strict_positive_single_interval",
            primary_include=True,
            final_rate=
                november_2021_row[
                    "release_rate_kg_h"
                ],
            selected_interval=
                november_2021_row[
                    "release_interval_id"
                ],
            conflict=False,
            notes=(
                "Single controlled-release interval. "
                "Sentinel-2 acquisition occurred "
                "inside the release window. Scene-level "
                "cloud percentage is near zero."
            ),
        )
    )

    # ---------------------------------------------------------
    # 2022-11-28:
    # retain as exploratory only.
    # ---------------------------------------------------------
    november_2022 = detail[
        detail["acquisition_date"].eq(
            "2022-11-28"
        )
    ]

    november_2022_row = require_one(
        november_2022,
        "2022-11-28 scene",
    )

    records.append(
        make_record(
            november_2022_row,
            review_status=
                "exploratory_conflicting_ultra_low",
            primary_include=False,
            final_rate=
                november_2022_row[
                    "release_rate_kg_h"
                ],
            selected_interval=
                november_2022_row[
                    "release_interval_id"
                ],
            conflict=True,
            notes=(
                "Meter-derived CH4 flow is approximately "
                "4.95 kg/h, but FacilityEmissionRate was "
                "reported as zero in another field. "
                "Retained only as an exploratory "
                "ultra-low-emission stress test."
            ),
        )
    )

    manifest = pd.DataFrame(
        records
    ).sort_values(
        "acquisition_time_utc"
    ).reset_index(drop=True)

    primary = manifest[
        manifest["primary_include"].eq(True)
    ].copy()

    manifest.to_csv(
        OUTPUT,
        index=False,
    )

    primary.to_csv(
        PRIMARY_OUTPUT,
        index=False,
    )

    print("=" * 110)
    print("SENTINEL-2 LOW-EMISSION SCENE MANIFEST V1")
    print("=" * 110)

    print(
        "\nAll unique scenes:",
        len(manifest),
    )

    print(
        "Primary strict scenes:",
        len(primary),
    )

    print(
        "Exploratory scenes:",
        int(
            (~manifest[
                "primary_include"
            ]).sum()
        ),
    )

    print("\nPrimary release rates:")
    print(
        primary[
            [
                "site",
                "acquisition_time_utc",
                "final_release_rate_kg_h",
                "final_emission_bin",
                "scene_cloud_percentage",
                "review_status",
            ]
        ].to_string(
            index=False,
        )
    )

    print("\nExcluded superseded intervals:")
    print(
        manifest[
            [
                "acquisition_time_utc",
                "excluded_release_interval_ids",
            ]
        ].to_string(
            index=False,
        )
    )

    print("\nSaved:")
    print(OUTPUT)
    print(PRIMARY_OUTPUT)


if __name__ == "__main__":
    main()
