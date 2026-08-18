from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform


BENCHMARK_INPUT = Path(
    "outputs/327_s2_low_emission_pilot_benchmark_v1.csv"
)

METRIC_OUTPUT = Path(
    "outputs/330_s2_low_emission_spatial_anomaly_metrics.csv"
)

SUMMARY_OUTPUT = Path(
    "outputs/331_s2_low_emission_spatial_anomaly_summary.csv"
)

MAP_DIR = Path(
    "outputs/s2_low_emission_spatial_anomaly_maps"
)


# GeoTIFF band order
BAND_INDEX = {
    "B2": 0,
    "B3": 1,
    "B4": 2,
    "B8": 3,
    "B8A": 4,
    "B11": 5,
    "B12": 6,
    "SCL": 7,
}


RADII_METERS = [
    100,
    200,
    500,
]


BAD_SCL = [
    3,
    8,
    9,
    10,
    11,
]


def read_reference(path):
    with rasterio.open(path) as dataset:
        array = dataset.read().astype(
            np.float64
        )

        profile = {
            "transform": dataset.transform,
            "crs": dataset.crs,
            "height": dataset.height,
            "width": dataset.width,
            "bounds": dataset.bounds,
        }

    return array, profile


def read_to_reference(
    path,
    reference_profile,
):
    with rasterio.open(path) as dataset:
        source = dataset.read().astype(
            np.float64
        )

        if (
            dataset.crs
            == reference_profile["crs"]
            and dataset.transform
            == reference_profile["transform"]
            and dataset.width
            == reference_profile["width"]
            and dataset.height
            == reference_profile["height"]
        ):
            return source

        destination = np.full(
            (
                dataset.count,
                reference_profile["height"],
                reference_profile["width"],
            ),
            np.nan,
            dtype=np.float64,
        )

        for band_index in range(
            dataset.count
        ):
            resampling = (
                Resampling.nearest
                if band_index
                == BAND_INDEX["SCL"]
                else Resampling.bilinear
            )

            reproject(
                source=source[band_index],
                destination=destination[
                    band_index
                ],
                src_transform=dataset.transform,
                src_crs=dataset.crs,
                dst_transform=reference_profile[
                    "transform"
                ],
                dst_crs=reference_profile[
                    "crs"
                ],
                dst_nodata=np.nan,
                resampling=resampling,
            )

        return destination


def calculate_indices(array):
    b8a = array[
        BAND_INDEX["B8A"]
    ]

    b11 = array[
        BAND_INDEX["B11"]
    ]

    b12 = array[
        BAND_INDEX["B12"]
    ]

    scl = np.rint(
        array[
            BAND_INDEX["SCL"]
        ]
    ).astype(int)

    valid = (
        np.isfinite(b8a)
        & np.isfinite(b11)
        & np.isfinite(b12)
        & (b8a > 0)
        & (b11 > 0)
        & (b12 > 0)
        & (scl != 0)
        & ~np.isin(
            scl,
            BAD_SCL,
        )
    )

    epsilon = 1e-6

    indices = {
        "B11":
            b11,

        "B12":
            b12,

        "B12_B11_ratio":
            b12 / (
                b11 + epsilon
            ),

        "B12_B11_nd":
            (
                b12 - b11
            ) / (
                b12 + b11 + epsilon
            ),

        "B11_B8A_ratio":
            b11 / (
                b8a + epsilon
            ),

        "B12_B8A_ratio":
            b12 / (
                b8a + epsilon
            ),
    }

    return indices, valid


def source_distance_grid(
    profile,
    latitude,
    longitude,
):
    transformed_x, transformed_y = (
        transform(
            "EPSG:4326",
            profile["crs"],
            [longitude],
            [latitude],
        )
    )

    source_x = transformed_x[0]
    source_y = transformed_y[0]

    rows, columns = np.indices(
        (
            profile["height"],
            profile["width"],
        )
    )

    xs, ys = rasterio.transform.xy(
        profile["transform"],
        rows,
        columns,
        offset="center",
    )

    # rasterio.transform.xy 可能會把二維 index
    # 攤平成一維，因此恢復成影像的 height × width。
    xs = np.asarray(
        xs,
        dtype=np.float64,
    ).reshape(rows.shape)

    ys = np.asarray(
        ys,
        dtype=np.float64,
    ).reshape(rows.shape)

    distance = np.sqrt(
        (xs - source_x) ** 2
        + (ys - source_y) ** 2
    )

    source_row, source_col = (
        rasterio.transform.rowcol(
            profile["transform"],
            source_x,
            source_y,
        )
    )

    return (
        distance,
        source_row,
        source_col,
    )


def anomaly_metrics(
    anomaly,
    valid,
    distance,
    radius,
):
    mask = (
        valid
        & np.isfinite(anomaly)
        & (distance <= radius)
    )

    values = anomaly[mask]

    if values.size == 0:
        return {
            "pixel_count": 0,
            "mean_anomaly": np.nan,
            "median_anomaly": np.nan,
            "mean_absolute_anomaly": np.nan,
            "p95_absolute_anomaly": np.nan,
        }

    return {
        "pixel_count":
            int(values.size),

        "mean_anomaly":
            float(
                np.mean(values)
            ),

        "median_anomaly":
            float(
                np.median(values)
            ),

        "mean_absolute_anomaly":
            float(
                np.mean(
                    np.abs(values)
                )
            ),

        "p95_absolute_anomaly":
            float(
                np.percentile(
                    np.abs(values),
                    95,
                )
            ),
    }


def save_anomaly_map(
    anomaly,
    valid,
    source_row,
    source_col,
    title,
    output_path,
):
    display = anomaly.copy()

    display[~valid] = np.nan

    finite_values = np.abs(
        display[
            np.isfinite(display)
        ]
    )

    if finite_values.size:
        limit = np.percentile(
            finite_values,
            98,
        )
    else:
        limit = 1.0

    if (
        not np.isfinite(limit)
        or limit == 0
    ):
        limit = 1.0

    plt.figure(
        figsize=(7, 6)
    )

    plt.imshow(
        display,
        vmin=-limit,
        vmax=limit,
        cmap="RdBu_r",
    )

    plt.scatter(
        [source_col],
        [source_row],
        marker="*",
        s=120,
        edgecolors="black",
        label="Release source",
    )

    plt.colorbar(
        label="Positive − negative median"
    )

    plt.title(title)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=180,
    )

    plt.close()


def main():
    benchmark = pd.read_csv(
        BENCHMARK_INPUT,
        low_memory=False,
    )

    benchmark["label"] = pd.to_numeric(
        benchmark["label"],
        errors="coerce",
    )

    benchmark["lat"] = pd.to_numeric(
        benchmark["lat"],
        errors="coerce",
    )

    benchmark["lon"] = pd.to_numeric(
        benchmark["lon"],
        errors="coerce",
    )

    MAP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metric_rows = []

    print("=" * 110)
    print(
        "SENTINEL-2 SPATIAL ANOMALY ANALYSIS"
    )
    print("=" * 110)

    for group_id, group in benchmark.groupby(
        "matched_group_id"
    ):
        positive = group[
            group["label"].eq(1)
        ]

        negatives = group[
            group["label"].eq(0)
        ]

        if (
            len(positive) != 1
            or len(negatives) != 4
        ):
            print(
                "Skipping invalid group:",
                group_id,
            )
            continue

        positive_row = positive.iloc[0]

        reference_array, profile = (
            read_reference(
                positive_row["patch_path"]
            )
        )

        distance, source_row, source_col = (
            source_distance_grid(
                profile,
                float(
                    positive_row["lat"]
                ),
                float(
                    positive_row["lon"]
                ),
            )
        )

        scene_arrays = {
            positive_row["sample_id"]:
                reference_array
        }

        scene_metadata = {
            positive_row["sample_id"]:
                positive_row
        }

        for _, negative_row in (
            negatives.iterrows()
        ):
            scene_arrays[
                negative_row["sample_id"]
            ] = read_to_reference(
                negative_row["patch_path"],
                profile,
            )

            scene_metadata[
                negative_row["sample_id"]
            ] = negative_row

        scene_indices = {}
        scene_valid_masks = {}

        for sample_id, array in (
            scene_arrays.items()
        ):
            indices, valid = (
                calculate_indices(
                    array
                )
            )

            scene_indices[
                sample_id
            ] = indices

            scene_valid_masks[
                sample_id
            ] = valid

        positive_id = (
            positive_row["sample_id"]
        )

        negative_ids = (
            negatives["sample_id"]
            .astype(str)
            .tolist()
        )

        for index_name in (
            scene_indices[
                positive_id
            ].keys()
        ):
            negative_stack = np.stack([
                scene_indices[
                    sample_id
                ][index_name]
                for sample_id
                in negative_ids
            ])

            negative_valid_stack = (
                np.stack([
                    scene_valid_masks[
                        sample_id
                    ]
                    for sample_id
                    in negative_ids
                ])
            )

            negative_stack[
                ~negative_valid_stack
            ] = np.nan

            negative_median = (
                np.nanmedian(
                    negative_stack,
                    axis=0,
                )
            )

            positive_values = (
                scene_indices[
                    positive_id
                ][index_name]
            )

            positive_valid = (
                scene_valid_masks[
                    positive_id
                ]
                & np.isfinite(
                    negative_median
                )
            )

            positive_anomaly = (
                positive_values
                - negative_median
            )

            for radius in (
                RADII_METERS
            ):
                metrics = anomaly_metrics(
                    positive_anomaly,
                    positive_valid,
                    distance,
                    radius,
                )

                metric_rows.append({
                    "matched_group_id":
                        group_id,

                    "sample_id":
                        positive_id,

                    "sample_role":
                        "positive",

                    "release_rate_kg_h":
                        positive_row[
                            "release_rate_kg_h"
                        ],

                    "index_name":
                        index_name,

                    "radius_m":
                        radius,

                    **metrics,
                })

            map_path = MAP_DIR / (
                f"{positive_id}_"
                f"{index_name}_anomaly.png"
            )

            save_anomaly_map(
                positive_anomaly,
                positive_valid,
                source_row,
                source_col,
                (
                    f"{positive_id}: {index_name}\n"
                    f"Positive − matched-negative median"
                ),
                map_path,
            )

            # Negative controls:
            # each negative compared with the other
            # three negatives.
            for control_id in (
                negative_ids
            ):
                reference_ids = [
                    sample_id
                    for sample_id
                    in negative_ids
                    if sample_id
                    != control_id
                ]

                control_reference_stack = (
                    np.stack([
                        scene_indices[
                            sample_id
                        ][index_name]
                        for sample_id
                        in reference_ids
                    ])
                )

                control_reference_valid = (
                    np.stack([
                        scene_valid_masks[
                            sample_id
                        ]
                        for sample_id
                        in reference_ids
                    ])
                )

                control_reference_stack[
                    ~control_reference_valid
                ] = np.nan

                control_median = (
                    np.nanmedian(
                        control_reference_stack,
                        axis=0,
                    )
                )

                control_values = (
                    scene_indices[
                        control_id
                    ][index_name]
                )

                control_valid = (
                    scene_valid_masks[
                        control_id
                    ]
                    & np.isfinite(
                        control_median
                    )
                )

                control_anomaly = (
                    control_values
                    - control_median
                )

                for radius in (
                    RADII_METERS
                ):
                    metrics = (
                        anomaly_metrics(
                            control_anomaly,
                            control_valid,
                            distance,
                            radius,
                        )
                    )

                    metric_rows.append({
                        "matched_group_id":
                            group_id,

                        "sample_id":
                            control_id,

                        "sample_role":
                            "negative_control",

                        "release_rate_kg_h":
                            0.0,

                        "index_name":
                            index_name,

                        "radius_m":
                            radius,

                        **metrics,
                    })

    metrics = pd.DataFrame(
        metric_rows
    )

    metrics.to_csv(
        METRIC_OUTPUT,
        index=False,
    )

    summary_rows = []

    group_columns = [
        "matched_group_id",
        "index_name",
        "radius_m",
    ]

    for keys, group in metrics.groupby(
        group_columns
    ):
        group_id, index_name, radius = keys

        positive = group[
            group["sample_role"].eq(
                "positive"
            )
        ]

        controls = group[
            group["sample_role"].eq(
                "negative_control"
            )
        ]

        if (
            len(positive) != 1
            or controls.empty
        ):
            continue

        positive_row = positive.iloc[0]

        for metric_name in [
            "mean_anomaly",
            "median_anomaly",
            "mean_absolute_anomaly",
            "p95_absolute_anomaly",
        ]:
            positive_value = float(
                positive_row[
                    metric_name
                ]
            )

            control_values = (
                pd.to_numeric(
                    controls[
                        metric_name
                    ],
                    errors="coerce",
                )
                .dropna()
            )

            if control_values.empty:
                continue

            absolute_rank = int(
                (
                    np.abs(
                        control_values
                    )
                    < abs(
                        positive_value
                    )
                ).sum()
                + 1
            )

            summary_rows.append({
                "matched_group_id":
                    group_id,

                "release_rate_kg_h":
                    positive_row[
                        "release_rate_kg_h"
                    ],

                "index_name":
                    index_name,

                "radius_m":
                    radius,

                "metric_name":
                    metric_name,

                "positive_value":
                    positive_value,

                "control_mean":
                    float(
                        control_values.mean()
                    ),

                "control_min":
                    float(
                        control_values.min()
                    ),

                "control_max":
                    float(
                        control_values.max()
                    ),

                "positive_absolute_rank_of_5":
                    absolute_rank,

                "positive_more_extreme_than_all_controls":
                    bool(
                        abs(positive_value)
                        > np.abs(
                            control_values
                        ).max()
                    ),
            })

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    strongest = summary[
        summary[
            "positive_more_extreme_than_all_controls"
        ].eq(True)
        & summary[
            "metric_name"
        ].isin([
            "mean_anomaly",
            "median_anomaly",
        ])
    ].copy()

    strongest = strongest.sort_values(
        [
            "matched_group_id",
            "radius_m",
            "index_name",
        ]
    )

    print("\nSpatial metric rows:", len(metrics))
    print("Summary rows:", len(summary))

    print(
        "\nCases where positive is more extreme "
        "than all four negative controls:"
    )

    if strongest.empty:
        print("None")
    else:
        print(
            strongest[
                [
                    "release_rate_kg_h",
                    "index_name",
                    "radius_m",
                    "metric_name",
                    "positive_value",
                    "control_min",
                    "control_max",
                    "positive_absolute_rank_of_5",
                ]
            ].to_string(
                index=False,
            )
        )

    print("\nSaved:")
    print(METRIC_OUTPUT)
    print(SUMMARY_OUTPUT)
    print(MAP_DIR)


if __name__ == "__main__":
    main()
