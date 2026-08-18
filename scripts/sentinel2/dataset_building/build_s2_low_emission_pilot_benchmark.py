from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as transform_coordinates


POSITIVE_MANIFEST = Path(
    "outputs/318_s2_low_emission_primary_scenes_v1.csv"
)

POSITIVE_QA = Path(
    "outputs/320_s2_low_emission_local_qa_v1.csv"
)

NEGATIVE_MANIFEST = Path(
    "outputs/324_s2_low_emission_matched_negative_manifest_v2.csv"
)

NEGATIVE_QA = Path(
    "outputs/326_s2_low_emission_negative_local_qa_v2.csv"
)


BENCHMARK_OUTPUT = Path(
    "outputs/327_s2_low_emission_pilot_benchmark_v1.csv"
)

FEATURE_OUTPUT = Path(
    "outputs/328_s2_low_emission_pilot_features_v1.csv"
)

ANOMALY_OUTPUT = Path(
    "outputs/329_s2_low_emission_matched_anomaly_features_v1.csv"
)


BANDS = [
    "B2",
    "B3",
    "B4",
    "B8",
    "B8A",
    "B11",
    "B12",
    "SCL",
]

REFLECTANCE_BANDS = BANDS[:-1]

LOCAL_HALF_SIZE_METERS = 500

BAD_SCL_CLASSES = [
    3,   # cloud shadow
    8,   # medium-probability cloud
    9,   # high-probability cloud
    10,  # thin cirrus
    11,  # snow/ice
]


def parse_bool_series(series):
    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def require_columns(frame, columns, name):
    missing = [
        column
        for column in columns
        if column not in frame.columns
    ]

    if missing:
        raise KeyError(
            f"{name} 缺少欄位："
            + ", ".join(missing)
        )


def load_positive_rows():
    manifest = pd.read_csv(
        POSITIVE_MANIFEST,
        low_memory=False,
    )

    qa = pd.read_csv(
        POSITIVE_QA,
        low_memory=False,
    )

    require_columns(
        manifest,
        [
            "scene_id",
            "site",
            "acquisition_time_utc",
            "lat",
            "lon",
            "final_release_rate_kg_h",
        ],
        "Positive manifest",
    )

    require_columns(
        qa,
        [
            "scene_id",
            "patch_path",
            "qa_pass_preliminary",
        ],
        "Positive QA",
    )

    selected_scene_ids = set(
        manifest["scene_id"]
        .dropna()
        .astype(str)
    )

    qa = qa[
        qa["scene_id"]
        .astype(str)
        .isin(selected_scene_ids)
    ].copy()

    qa = qa.drop_duplicates(
        subset=["scene_id"],
        keep="first",
    )

    positive = manifest.merge(
        qa[
            [
                "scene_id",
                "patch_path",
                "qa_pass_preliminary",
                "local_valid_fraction",
                "local_cloud_fraction",
                "local_shadow_fraction",
                "local_bad_atmosphere_fraction",
                "local_all_zero_fraction",
            ]
        ],
        on="scene_id",
        how="left",
        validate="one_to_one",
    )

    positive = positive.sort_values(
        "acquisition_time_utc"
    ).reset_index(drop=True)

    positive["sample_id"] = [
        f"S2_LOW_POS_{number:02d}"
        for number in range(
            1,
            len(positive) + 1,
        )
    ]

    positive["label"] = 1
    positive["dataset_role"] = (
        "strict_low_emission_positive"
    )

    positive["matched_group_id"] = (
        positive["scene_id"]
        .astype(str)
    )

    positive["matched_positive_scene_id"] = (
        positive["scene_id"]
    )

    positive["release_rate_kg_h"] = (
        pd.to_numeric(
            positive[
                "final_release_rate_kg_h"
            ],
            errors="coerce",
        )
    )

    return positive


def load_negative_rows():
    manifest = pd.read_csv(
        NEGATIVE_MANIFEST,
        low_memory=False,
    )

    qa = pd.read_csv(
        NEGATIVE_QA,
        low_memory=False,
    )

    require_columns(
        manifest,
        [
            "negative_id",
            "scene_id",
            "site",
            "acquisition_time_utc",
            "lat",
            "lon",
            "matched_positive_scene_id",
        ],
        "Negative manifest",
    )

    require_columns(
        qa,
        [
            "scene_id",
            "patch_path",
            "qa_pass_preliminary",
        ],
        "Negative QA",
    )

    qa = qa.drop_duplicates(
        subset=["scene_id"],
        keep="first",
    )

    negative = manifest.merge(
        qa[
            [
                "scene_id",
                "patch_path",
                "qa_pass_preliminary",
                "local_valid_fraction",
                "local_cloud_fraction",
                "local_shadow_fraction",
                "local_bad_atmosphere_fraction",
                "local_all_zero_fraction",
            ]
        ],
        on="scene_id",
        how="left",
        validate="one_to_one",
    )

    negative["sample_id"] = (
        negative["negative_id"]
        .astype(str)
    )

    negative["label"] = 0
    negative["dataset_role"] = (
        "clean_matched_negative"
    )

    negative["matched_group_id"] = (
        negative[
            "matched_positive_scene_id"
        ].astype(str)
    )

    negative["release_rate_kg_h"] = 0.0

    return negative


def build_benchmark():
    positive = load_positive_rows()
    negative = load_negative_rows()

    common_columns = sorted(
        set(positive.columns)
        | set(negative.columns)
    )

    positive = positive.reindex(
        columns=common_columns
    )

    negative = negative.reindex(
        columns=common_columns
    )

    benchmark = pd.concat(
        [positive, negative],
        ignore_index=True,
        sort=False,
    )

    benchmark[
        "acquisition_time_utc"
    ] = pd.to_datetime(
        benchmark[
            "acquisition_time_utc"
        ],
        errors="coerce",
        utc=True,
    )

    benchmark["lat"] = pd.to_numeric(
        benchmark["lat"],
        errors="coerce",
    )

    benchmark["lon"] = pd.to_numeric(
        benchmark["lon"],
        errors="coerce",
    )

    benchmark[
        "qa_pass_preliminary"
    ] = parse_bool_series(
        benchmark[
            "qa_pass_preliminary"
        ]
    )

    benchmark["patch_exists"] = (
        benchmark["patch_path"]
        .fillna("")
        .map(
            lambda value:
                Path(value).exists()
                if str(value).strip()
                else False
        )
    )

    benchmark[
        "benchmark_version"
    ] = "s2_low_emission_pilot_v1"

    benchmark = benchmark.sort_values(
        [
            "matched_group_id",
            "label",
            "acquisition_time_utc",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    ).reset_index(drop=True)

    if len(benchmark) != 10:
        raise RuntimeError(
            "預期 benchmark 有 10 張影像，"
            f"實際找到 {len(benchmark)} 張。"
        )

    if benchmark["label"].value_counts().to_dict() != {
        0: 8,
        1: 2,
    }:
        raise RuntimeError(
            "標籤數量不正確："
            + str(
                benchmark[
                    "label"
                ].value_counts().to_dict()
            )
        )

    if not benchmark[
        "qa_pass_preliminary"
    ].all():
        failed = benchmark[
            ~benchmark[
                "qa_pass_preliminary"
            ]
        ]

        raise RuntimeError(
            "仍有影像未通過 QA：\n"
            + failed[
                [
                    "sample_id",
                    "scene_id",
                ]
            ].to_string(index=False)
        )

    if not benchmark[
        "patch_exists"
    ].all():
        missing = benchmark[
            ~benchmark["patch_exists"]
        ]

        raise FileNotFoundError(
            "找不到部分 patch：\n"
            + missing[
                [
                    "sample_id",
                    "patch_path",
                ]
            ].to_string(index=False)
        )

    return benchmark


def numeric_statistics(values, prefix):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_p10": np.nan,
            f"{prefix}_p25": np.nan,
            f"{prefix}_p75": np.nan,
            f"{prefix}_p90": np.nan,
            f"{prefix}_p95": np.nan,
        }

    return {
        f"{prefix}_mean":
            float(np.mean(values)),

        f"{prefix}_median":
            float(np.median(values)),

        f"{prefix}_std":
            float(np.std(values)),

        f"{prefix}_p10":
            float(np.percentile(
                values,
                10,
            )),

        f"{prefix}_p25":
            float(np.percentile(
                values,
                25,
            )),

        f"{prefix}_p75":
            float(np.percentile(
                values,
                75,
            )),

        f"{prefix}_p90":
            float(np.percentile(
                values,
                90,
            )),

        f"{prefix}_p95":
            float(np.percentile(
                values,
                95,
            )),
    }


def extract_patch_features(row):
    patch_path = Path(
        row["patch_path"]
    )

    latitude = float(row["lat"])
    longitude = float(row["lon"])

    with rasterio.open(
        patch_path
    ) as dataset:
        array = dataset.read().astype(
            np.float64
        )

        if dataset.count != len(BANDS):
            raise RuntimeError(
                f"{patch_path}: 預期 "
                f"{len(BANDS)} bands，"
                f"實際為 {dataset.count}。"
            )

        transformed_x, transformed_y = (
            transform_coordinates(
                "EPSG:4326",
                dataset.crs,
                [longitude],
                [latitude],
            )
        )

        center_row, center_col = (
            dataset.index(
                transformed_x[0],
                transformed_y[0],
            )
        )

        pixel_size = float(
            np.mean([
                abs(dataset.res[0]),
                abs(dataset.res[1]),
            ])
        )

        radius_pixels = max(
            1,
            int(
                np.ceil(
                    LOCAL_HALF_SIZE_METERS
                    / pixel_size
                )
            ),
        )

        row_start = max(
            0,
            center_row - radius_pixels,
        )

        row_end = min(
            dataset.height,
            center_row + radius_pixels + 1,
        )

        col_start = max(
            0,
            center_col - radius_pixels,
        )

        col_end = min(
            dataset.width,
            center_col + radius_pixels + 1,
        )

        local = array[
            :,
            row_start:row_end,
            col_start:col_end,
        ]

        reflectance = local[
            :len(REFLECTANCE_BANDS)
        ]

        scl = np.rint(
            local[
                BANDS.index("SCL")
            ]
        ).astype(int)

        finite = np.all(
            np.isfinite(reflectance),
            axis=0,
        )

        nonzero = ~np.all(
            reflectance == 0,
            axis=0,
        )

        atmosphere_good = ~np.isin(
            scl,
            BAD_SCL_CLASSES,
        )

        scl_valid = scl != 0

        valid = (
            finite
            & nonzero
            & atmosphere_good
            & scl_valid
        )

        if valid.sum() == 0:
            raise RuntimeError(
                f"{patch_path}: local window "
                "沒有有效像素。"
            )

        feature = {
            "sample_id":
                row["sample_id"],

            "scene_id":
                row["scene_id"],

            "matched_group_id":
                row[
                    "matched_group_id"
                ],

            "site":
                row["site"],

            "acquisition_time_utc":
                row[
                    "acquisition_time_utc"
                ],

            "label":
                int(row["label"]),

            "dataset_role":
                row["dataset_role"],

            "release_rate_kg_h":
                float(
                    row[
                        "release_rate_kg_h"
                    ]
                ),

            "patch_path":
                str(patch_path),

            "local_pixel_count":
                int(valid.size),

            "local_valid_pixel_count":
                int(valid.sum()),

            "local_valid_fraction_recomputed":
                float(valid.mean()),

            "pixel_size_m":
                pixel_size,
        }

        band_arrays = {}

        for band_index, band in enumerate(
            REFLECTANCE_BANDS
        ):
            band_values = reflectance[
                band_index
            ]

            band_arrays[band] = (
                band_values
            )

            feature.update(
                numeric_statistics(
                    band_values[valid],
                    f"local_{band}",
                )
            )

        epsilon = 1e-6

        b8a = band_arrays["B8A"]
        b11 = band_arrays["B11"]
        b12 = band_arrays["B12"]

        index_arrays = {
            "swir_ratio_B12_B11":
                b12 / (
                    b11 + epsilon
                ),

            "swir_nd_B12_B11":
                (
                    b12 - b11
                ) / (
                    b12 + b11 + epsilon
                ),

            "ratio_B11_B8A":
                b11 / (
                    b8a + epsilon
                ),

            "ratio_B12_B8A":
                b12 / (
                    b8a + epsilon
                ),

            "ndmi_B8A_B11":
                (
                    b8a - b11
                ) / (
                    b8a + b11 + epsilon
                ),

            "swir_difference_B12_B11":
                b12 - b11,
        }

        for index_name, index_array in (
            index_arrays.items()
        ):
            feature.update(
                numeric_statistics(
                    index_array[valid],
                    f"local_{index_name}",
                )
            )

        # 釋放點附近約 5×5 pixels。
        source_radius = 2

        source_row_start = max(
            0,
            center_row - source_radius,
        )

        source_row_end = min(
            dataset.height,
            center_row + source_radius + 1,
        )

        source_col_start = max(
            0,
            center_col - source_radius,
        )

        source_col_end = min(
            dataset.width,
            center_col + source_radius + 1,
        )

        source_window = array[
            :,
            source_row_start:source_row_end,
            source_col_start:source_col_end,
        ]

        source_reflectance = source_window[
            :len(REFLECTANCE_BANDS)
        ]

        source_scl = np.rint(
            source_window[
                BANDS.index("SCL")
            ]
        ).astype(int)

        source_valid = (
            np.all(
                np.isfinite(
                    source_reflectance
                ),
                axis=0,
            )
            & ~np.all(
                source_reflectance == 0,
                axis=0,
            )
            & ~np.isin(
                source_scl,
                BAD_SCL_CLASSES,
            )
            & (source_scl != 0)
        )

        for band_index, band in enumerate(
            REFLECTANCE_BANDS
        ):
            values = source_reflectance[
                band_index
            ][source_valid]

            feature[
                f"source5x5_{band}_mean"
            ] = (
                float(np.mean(values))
                if values.size
                else np.nan
            )

            feature[
                f"source5x5_{band}_median"
            ] = (
                float(np.median(values))
                if values.size
                else np.nan
            )

        return feature


def create_matched_anomalies(features):
    metadata_columns = {
        "sample_id",
        "scene_id",
        "matched_group_id",
        "site",
        "acquisition_time_utc",
        "label",
        "dataset_role",
        "release_rate_kg_h",
        "patch_path",
    }

    feature_columns = [
        column
        for column in features.columns
        if (
            column not in metadata_columns
            and pd.api.types.is_numeric_dtype(
                features[column]
            )
        )
    ]

    result = features.copy()

    for feature_column in feature_columns:
        result[
            f"{feature_column}__neg_mean"
        ] = np.nan

        result[
            f"{feature_column}__anomaly"
        ] = np.nan

        result[
            f"{feature_column}__z_vs_neg"
        ] = np.nan

    for group_id, group in (
        features.groupby(
            "matched_group_id"
        )
    ):
        negatives = group[
            group["label"].eq(0)
        ]

        if negatives.empty:
            continue

        group_mask = result[
            "matched_group_id"
        ].eq(group_id)

        for feature_column in feature_columns:
            negative_values = pd.to_numeric(
                negatives[
                    feature_column
                ],
                errors="coerce",
            ).dropna()

            if negative_values.empty:
                continue

            negative_mean = float(
                negative_values.mean()
            )

            negative_std = float(
                negative_values.std(
                    ddof=0
                )
            )

            result.loc[
                group_mask,
                f"{feature_column}__neg_mean",
            ] = negative_mean

            result.loc[
                group_mask,
                f"{feature_column}__anomaly",
            ] = (
                pd.to_numeric(
                    result.loc[
                        group_mask,
                        feature_column,
                    ],
                    errors="coerce",
                )
                - negative_mean
            )

            if negative_std > 0:
                result.loc[
                    group_mask,
                    f"{feature_column}__z_vs_neg",
                ] = (
                    result.loc[
                        group_mask,
                        f"{feature_column}__anomaly",
                    ]
                    / negative_std
                )

    return result


def main():
    benchmark = build_benchmark()

    benchmark.to_csv(
        BENCHMARK_OUTPUT,
        index=False,
    )

    print("=" * 110)
    print(
        "SENTINEL-2 LOW-EMISSION "
        "PILOT BENCHMARK V1"
    )
    print("=" * 110)

    print(
        "\nBenchmark samples:",
        len(benchmark),
    )

    print("\nLabels:")
    print(
        benchmark["label"]
        .value_counts()
        .sort_index()
    )

    print("\nSamples per matched group:")
    print(
        benchmark.groupby(
            "matched_group_id"
        )["label"].agg(
            total="size",
            positives="sum",
        )
    )

    print("\nExtracting features...")

    feature_rows = []

    for number, row in (
        benchmark.iterrows()
    ):
        print(
            f"[{number + 1}/{len(benchmark)}] "
            f"{row['sample_id']} | "
            f"label={row['label']}",
            flush=True,
        )

        feature_rows.append(
            extract_patch_features(row)
        )

    features = pd.DataFrame(
        feature_rows
    )

    features.to_csv(
        FEATURE_OUTPUT,
        index=False,
    )

    anomaly = create_matched_anomalies(
        features
    )

    anomaly.to_csv(
        ANOMALY_OUTPUT,
        index=False,
    )

    print("\n" + "=" * 110)
    print("FEATURE EXTRACTION SUMMARY")
    print("=" * 110)

    print(
        "\nFeature rows:",
        len(features),
    )

    print(
        "Feature columns:",
        len(features.columns),
    )

    print(
        "Missing numeric values:",
        int(
            features.select_dtypes(
                include=[np.number]
            ).isna().sum().sum()
        ),
    )

    important_features = [
        "local_B11_median",
        "local_B12_median",
        "local_swir_ratio_B12_B11_median",
        "local_swir_nd_B12_B11_median",
        "local_ratio_B11_B8A_median",
        "local_ratio_B12_B8A_median",
    ]

    important_features = [
        column
        for column in important_features
        if column in anomaly.columns
    ]

    positives = anomaly[
        anomaly["label"].eq(1)
    ].copy()

    print(
        "\nPositive matched-negative anomalies:"
    )

    display_columns = [
        "sample_id",
        "release_rate_kg_h",
        "matched_group_id",
    ]

    for column in important_features:
        display_columns.extend([
            column,
            f"{column}__neg_mean",
            f"{column}__anomaly",
            f"{column}__z_vs_neg",
        ])

    print(
        positives[
            display_columns
        ].to_string(
            index=False,
        )
    )

    print("\nSaved:")
    print(BENCHMARK_OUTPUT)
    print(FEATURE_OUTPUT)
    print(ANOMALY_OUTPUT)


if __name__ == "__main__":
    main()
