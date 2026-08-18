from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_DIR = Path("outputs")

S2_LOW_INPUT = Path(
    "outputs/341_s2_low_emission_pilot_v1_locked.csv"
)

S2_HIGH_INPUT = Path(
    "outputs/372_s2_high_emission_benchmark_v1.csv"
)

S2_LOW_RESULT_INPUT = Path(
    "outputs/347_s2_locked_wind_aligned_scene_conclusions_v1.csv"
)

S2_HIGH_RESULT_INPUT = Path(
    "outputs/387_s2_high_emission_wind_aligned_scene_conclusions_v1.csv"
)

MASTER_OUTPUT = Path(
    "outputs/390_multisensor_master_manifest_v1.csv"
)

GAP_OUTPUT = Path(
    "outputs/391_multisensor_dataset_gap_summary_v1.csv"
)

REPORT_OUTPUT = Path(
    "outputs/392_multisensor_benchmark_report_v1.txt"
)


def parse_bool(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin([
            "true",
            "1",
            "yes",
        ])
    )


def first_column(
    frame,
    candidates,
    required=False,
):
    for column in candidates:
        if column in frame.columns:
            return column

    if required:
        raise KeyError(
            "找不到任何候選欄位："
            + ", ".join(candidates)
        )

    return None


def find_landsat_confirmed_file():
    candidates = sorted(
        OUTPUT_DIR.glob(
            "*landsat*.csv"
        )
    )

    matches = []

    for path in candidates:
        try:
            frame = pd.read_csv(
                path,
                low_memory=False,
            )
        except Exception:
            continue

        required = {
            "label",
            "landsat_sensor",
        }

        if (
            len(frame) == 9
            and required.issubset(
                frame.columns
            )
        ):
            score = 0

            for column in [
                "final_scene_label",
                "final_label_source",
                "review_status",
                "recommended_label",
                "raster_group_id",
                "pixel_hash",
            ]:
                if column in frame.columns:
                    score += 1

            matches.append(
                (
                    score,
                    path.stat().st_mtime,
                    path,
                    frame,
                )
            )

    if not matches:
        raise FileNotFoundError(
            "找不到具有 9 筆、包含 label 與 "
            "landsat_sensor 的最終 Landsat CSV。"
        )

    matches.sort(
        key=lambda item: (
            item[0],
            item[1],
        ),
        reverse=True,
    )

    score, _, path, frame = (
        matches[0]
    )

    print(
        "Selected Landsat file:",
        path,
    )

    print(
        "Landsat selection score:",
        score,
    )

    return path, frame


def load_wind_status(
    path,
):
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "matched_group_id",
                "wind_aligned_status",
                "wind_evidence_tier",
            ]
        )

    frame = pd.read_csv(
        path,
        low_memory=False,
    )

    group_column = first_column(
        frame,
        [
            "matched_group_id",
            "positive_scene_id",
            "scene_id",
        ],
        required=True,
    )

    status_column = first_column(
        frame,
        [
            "final_wind_aligned_status",
            "locked_scene_conclusion",
        ],
        required=True,
    )

    result = pd.DataFrame({
        "matched_group_id":
            frame[
                group_column
            ].astype(str),

        "wind_aligned_status":
            frame[
                status_column
            ].astype(str),
    })

    if (
        "wind_evidence_tier"
        in frame.columns
    ):
        result[
            "wind_evidence_tier"
        ] = frame[
            "wind_evidence_tier"
        ].astype(str)
    else:
        result[
            "wind_evidence_tier"
        ] = "primary"

    return result.drop_duplicates(
        subset=["matched_group_id"],
        keep="first",
    )


def standardize_sentinel2(
    frame,
    dataset_name,
    emission_category,
    result_path,
):
    required_columns = [
        "sample_id",
        "scene_id",
        "acquisition_time_utc",
        "site",
        "label",
        "release_rate_kg_h",
        "matched_group_id",
        "patch_path",
        "qa_pass_preliminary",
    ]

    missing = [
        column
        for column in required_columns
        if column not in frame.columns
    ]

    if missing:
        raise KeyError(
            f"{dataset_name} 缺少欄位："
            + ", ".join(missing)
        )

    result = pd.DataFrame({
        "sample_id":
            frame[
                "sample_id"
            ].astype(str),

        "sensor_family":
            "Sentinel-2 MSI",

        "platform":
            (
                frame[
                    "spacecraft_name"
                ].fillna(
                    "Sentinel-2"
                ).astype(str)
                if "spacecraft_name"
                in frame.columns
                else "Sentinel-2"
            ),

        "scene_id":
            frame[
                "scene_id"
            ].astype(str),

        "site":
            frame[
                "site"
            ].astype(str),

        "acquisition_time_utc":
            pd.to_datetime(
                frame[
                    "acquisition_time_utc"
                ],
                errors="coerce",
                utc=True,
            ),

        "label":
            pd.to_numeric(
                frame["label"],
                errors="coerce",
            ),

        "release_rate_kg_h":
            pd.to_numeric(
                frame[
                    "release_rate_kg_h"
                ],
                errors="coerce",
            ),

        "matched_group_id":
            frame[
                "matched_group_id"
            ].astype(str),

        "patch_path":
            frame[
                "patch_path"
            ].astype(str),

        "qa_pass":
            parse_bool(
                frame[
                    "qa_pass_preliminary"
                ]
            ),

        "patch_exists":
            (
                parse_bool(
                    frame[
                        "patch_exists"
                    ]
                )
                if "patch_exists"
                in frame.columns
                else frame[
                    "patch_path"
                ].astype(str).map(
                    lambda value:
                        Path(value).exists()
                )
            ),

        "dataset_name":
            dataset_name,

        "emission_category":
            emission_category,

        "benchmark_design":
            "matched_case_control",

        "benchmark_ready":
            True,

        "benchmark_limitation":
            (
                "small matched controlled-release "
                "benchmark"
            ),

        "ground_truth_status":
            "strict_controlled_release",

        "source_manifest":
            (
                str(S2_LOW_INPUT)
                if emission_category
                == "low_emission"
                else str(S2_HIGH_INPUT)
            ),
    })

    if (
        "matched_positive_rate_kg_h"
        in frame.columns
    ):
        result[
            "matched_positive_release_rate_kg_h"
        ] = pd.to_numeric(
            frame[
                "matched_positive_rate_kg_h"
            ],
            errors="coerce",
        )
    else:
        result[
            "matched_positive_release_rate_kg_h"
        ] = np.where(
            result["label"].eq(1),
            result[
                "release_rate_kg_h"
            ],
            np.nan,
        )

    wind = load_wind_status(
        result_path
    )

    result = result.merge(
        wind,
        on="matched_group_id",
        how="left",
        validate="many_to_one",
    )

    return result


def standardize_landsat(
    frame,
    source_path,
):
    scene_column = first_column(
        frame,
        [
            "scene_id",
            "landsat_scene_id",
            "landsat_image_id",
            "image_id",
            "system_index",
        ],
    )

    sample_column = first_column(
        frame,
        [
            "raster_group_id",
            "sample_id",
            "pixel_hash",
            "patch_id",
        ],
    )

    time_column = first_column(
        frame,
        [
            "landsat_image_time",
            "acquisition_time_utc",
            "image_time_utc",
            "scene_time_utc",
        ],
        required=True,
    )

    site_column = first_column(
        frame,
        [
            "site",
            "site_name",
            "release_site",
        ],
        required=True,
    )

    rate_column = first_column(
        frame,
        [
            "release_rate_kg_h",
            "preferred_release_rate_kg_h",
            "final_release_rate_kg_h",
            "matched_release_rate_kg_h",
            "cr_kgh_CH4_mean300",
        ],
    )

    patch_column = first_column(
        frame,
        [
            "patch_path",
            "raster_path",
            "file_path",
            "tif_path",
            "filename",
        ],
    )

    if sample_column is None:
        sample_ids = [
            f"LANDSAT_CONFIRMED_{number:02d}"
            for number in range(
                1,
                len(frame) + 1,
            )
        ]
    else:
        sample_ids = (
            frame[
                sample_column
            ].astype(str)
        )

    if scene_column is None:
        scene_ids = (
            pd.Series(
                sample_ids,
                index=frame.index,
            )
        )
    else:
        scene_ids = (
            frame[
                scene_column
            ].astype(str)
        )

    if rate_column is None:
        release_rates = pd.Series(
            np.nan,
            index=frame.index,
        )
    else:
        release_rates = (
            pd.to_numeric(
                frame[
                    rate_column
                ],
                errors="coerce",
            )
        )

    if patch_column is None:
        patch_paths = pd.Series(
            pd.NA,
            index=frame.index,
            dtype="object",
        )
    else:
        patch_paths = (
            frame[
                patch_column
            ].astype(str)
        )

    if (
        "qa_pass_preliminary"
        in frame.columns
    ):
        qa_pass = parse_bool(
            frame[
                "qa_pass_preliminary"
            ]
        )

    elif (
        "all_zero"
        in frame.columns
        and "has_nan"
        in frame.columns
    ):
        qa_pass = (
            ~parse_bool(
                frame["all_zero"]
            )
            & ~parse_bool(
                frame["has_nan"]
            )
        )

    else:
        qa_pass = pd.Series(
            True,
            index=frame.index,
        )

    if "patch_exists" in frame.columns:
        patch_exists = parse_bool(
            frame["patch_exists"]
        )
    elif patch_column is not None:
        patch_exists = (
            patch_paths.map(
                lambda value:
                    Path(str(value)).exists()
            )
        )
    else:
        patch_exists = pd.Series(
            False,
            index=frame.index,
        )

    result = pd.DataFrame({
        "sample_id":
            sample_ids,

        "sensor_family":
            "Landsat OLI",

        "platform":
            frame[
                "landsat_sensor"
            ].astype(str),

        "scene_id":
            scene_ids,

        "site":
            frame[
                site_column
            ].astype(str),

        "acquisition_time_utc":
            pd.to_datetime(
                frame[
                    time_column
                ],
                errors="coerce",
                utc=True,
            ),

        "label":
            pd.to_numeric(
                frame["label"],
                errors="coerce",
            ),

        "release_rate_kg_h":
            release_rates,

        "matched_positive_release_rate_kg_h":
            np.where(
                pd.to_numeric(
                    frame["label"],
                    errors="coerce",
                ).eq(1),
                release_rates,
                np.nan,
            ),

        "matched_group_id":
            pd.NA,

        "patch_path":
            patch_paths,

        "qa_pass":
            qa_pass,

        "patch_exists":
            patch_exists,

        "dataset_name":
            "landsat_confirmed_scene_set",

        "emission_category":
            "mixed_emission",

        "benchmark_design":
            "confirmed_scene_set_unmatched",

        "benchmark_ready":
            False,

        "benchmark_limitation":
            (
                "Only 2 confirmed negatives; "
                "matched negative expansion required"
            ),

        "ground_truth_status":
            "confirmed_release_interval_review",

        "source_manifest":
            str(source_path),

        "wind_aligned_status":
            pd.NA,

        "wind_evidence_tier":
            pd.NA,
    })

    return result


def build_gap_summary(
    master,
):
    rows = []

    group_columns = [
        "sensor_family",
        "platform",
        "benchmark_design",
    ]

    for keys, group in master.groupby(
        group_columns,
        dropna=False,
        sort=False,
    ):
        (
            sensor_family,
            platform,
            benchmark_design,
        ) = keys

        rows.append({
            "summary_type":
                "platform_dataset",

            "sensor_family":
                sensor_family,

            "platform":
                platform,

            "benchmark_design":
                benchmark_design,

            "scene_count":
                len(group),

            "positive_count":
                int(
                    group[
                        "label"
                    ].eq(1).sum()
                ),

            "negative_count":
                int(
                    group[
                        "label"
                    ].eq(0).sum()
                ),

            "qa_pass_count":
                int(
                    group[
                        "qa_pass"
                    ].eq(True).sum()
                ),

            "patch_exists_count":
                int(
                    group[
                        "patch_exists"
                    ].eq(True).sum()
                ),

            "unique_sites":
                int(
                    group[
                        "site"
                    ].nunique()
                ),

            "matched_group_count":
                int(
                    group[
                        "matched_group_id"
                    ].dropna().nunique()
                ),

            "benchmark_ready_scene_count":
                int(
                    group[
                        "benchmark_ready"
                    ].eq(True).sum()
                ),

            "acquisition_start":
                group[
                    "acquisition_time_utc"
                ].min(),

            "acquisition_end":
                group[
                    "acquisition_time_utc"
                ].max(),
        })

    summary = pd.DataFrame(
        rows
    )

    return summary


def main():
    low = pd.read_csv(
        S2_LOW_INPUT,
        low_memory=False,
    )

    high = pd.read_csv(
        S2_HIGH_INPUT,
        low_memory=False,
    )

    landsat_path, landsat = (
        find_landsat_confirmed_file()
    )

    low_standard = (
        standardize_sentinel2(
            frame=low,
            dataset_name=
                "s2_low_emission_pilot_v1",
            emission_category=
                "low_emission",
            result_path=
                S2_LOW_RESULT_INPUT,
        )
    )

    high_standard = (
        standardize_sentinel2(
            frame=high,
            dataset_name=
                "s2_high_emission_benchmark_v1",
            emission_category=
                "high_emission",
            result_path=
                S2_HIGH_RESULT_INPUT,
        )
    )

    landsat_standard = (
        standardize_landsat(
            frame=landsat,
            source_path=landsat_path,
        )
    )

    master = pd.concat(
        [
            low_standard,
            high_standard,
            landsat_standard,
        ],
        ignore_index=True,
        sort=False,
    )

    master["label"] = pd.to_numeric(
        master["label"],
        errors="raise",
    ).astype(int)

    master[
        "release_rate_kg_h"
    ] = pd.to_numeric(
        master[
            "release_rate_kg_h"
        ],
        errors="coerce",
    )

    master[
        "master_manifest_version"
    ] = "multisensor_master_v1"

    master = master.sort_values(
        [
            "sensor_family",
            "platform",
            "acquisition_time_utc",
            "label",
        ]
    ).reset_index(drop=True)

    duplicate_key = (
        master[
            [
                "sensor_family",
                "scene_id",
            ]
        ]
        .astype(str)
        .duplicated(
            keep=False
        )
    )

    master[
        "duplicate_scene_within_sensor"
    ] = duplicate_key

    master.to_csv(
        MASTER_OUTPUT,
        index=False,
    )

    gap = build_gap_summary(
        master
    )

    gap.to_csv(
        GAP_OUTPUT,
        index=False,
    )

    total_scenes = len(master)

    positive_count = int(
        master["label"].eq(1).sum()
    )

    negative_count = int(
        master["label"].eq(0).sum()
    )

    sensor_family_count = int(
        master[
            "sensor_family"
        ].nunique()
    )

    platform_count = int(
        master[
            "platform"
        ].nunique()
    )

    matched_family_count = int(
        master.loc[
            master[
                "benchmark_design"
            ].eq(
                "matched_case_control"
            ),
            "sensor_family",
        ].nunique()
    )

    required_sensor_families = 4

    family_gap = max(
        required_sensor_families
        - sensor_family_count,
        0,
    )

    report_lines = [
        "=" * 115,
        "MULTISENSOR CONTROLLED-RELEASE BENCHMARK REPORT V1",
        "=" * 115,
        "",
        f"Total scenes: {total_scenes}",
        f"Positive scenes: {positive_count}",
        f"Negative scenes: {negative_count}",
        "",
        f"Sensor families represented: {sensor_family_count}",
        f"Platforms represented: {platform_count}",
        (
            "Sensor families with matched "
            f"case-control benchmark: {matched_family_count}"
        ),
        "",
        (
            "Target sensor families requested: "
            f"{required_sensor_families}"
        ),
        (
            "Additional sensor families still needed: "
            f"{family_gap}"
        ),
        "",
        "Platform summary:",
        gap.to_string(index=False),
        "",
        "Important interpretation:",
        (
            "Sentinel-2 currently has a complete matched "
            "case-control benchmark."
        ),
        (
            "Landsat currently has a confirmed scene set "
            "with 7 positives and 2 negatives, but it is "
            "not yet a balanced matched benchmark."
        ),
        (
            "Landsat-8 and Landsat-9 are separate spacecraft "
            "but belong to the same Landsat OLI sensor family."
        ),
        (
            "Current controlled-release coverage therefore "
            "represents two sensor families, not four."
        ),
        "",
        "Next research gap:",
        (
            "Expand Landsat matched negatives and add at least "
            "two additional satellite sensor families with "
            "retrievable raw imagery."
        ),
        "",
        f"Selected Landsat input: {landsat_path}",
    ]

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("=" * 115)
    print("MULTISENSOR MASTER MANIFEST SUMMARY")
    print("=" * 115)

    print(
        "\nSelected Landsat file:",
        landsat_path,
    )

    print(
        "\nTotal scenes:",
        total_scenes,
    )

    print(
        "Positive scenes:",
        positive_count,
    )

    print(
        "Negative scenes:",
        negative_count,
    )

    print(
        "\nSensor families:",
        sensor_family_count,
    )

    print(
        "Platforms:",
        platform_count,
    )

    print(
        "Matched benchmark sensor families:",
        matched_family_count,
    )

    print(
        "Additional sensor families needed:",
        family_gap,
    )

    print("\nPlatform summary:")
    print(
        gap.to_string(
            index=False
        )
    )

    print(
        "\nDuplicated scene rows within sensor:",
        int(
            master[
                "duplicate_scene_within_sensor"
            ].sum()
        ),
    )

    print("\nSaved:")
    print(MASTER_OUTPUT)
    print(GAP_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
