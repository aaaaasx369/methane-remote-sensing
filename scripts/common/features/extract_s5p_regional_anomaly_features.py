from pathlib import Path
import os
import time

import ee
import numpy as np
import pandas as pd


INPUT = Path(
    "outputs/503_s5p_nearest_valid_orbit_manifest_v1.csv"
)

FEATURE_OUTPUT = Path(
    "outputs/506_s5p_regional_anomaly_features_v1.csv"
)

VALID_OUTPUT = Path(
    "outputs/507_s5p_regional_anomaly_valid_features_v1.csv"
)

REPORT_OUTPUT = Path(
    "outputs/508_s5p_regional_anomaly_report_v1.txt"
)


COLLECTION_ID = (
    "COPERNICUS/S5P/OFFL/L3_CH4"
)

CH4_BAND = (
    "CH4_column_volume_mixing_ratio_"
    "dry_air_bias_corrected"
)

UNCERTAINTY_BAND = (
    "CH4_column_volume_mixing_ratio_"
    "dry_air_uncertainty"
)

GRID_SCALE_M = 1113.2

SOURCE_RADIUS_M = 10000
BACKGROUND_INNER_RADIUS_M = 20000
BACKGROUND_OUTER_RADIUS_M = 50000

REQUEST_DELAY_SECONDS = 0.15


def find_column(frame, candidates, description):
    for column in candidates:
        if column in frame.columns:
            return column

    raise KeyError(
        f"Cannot find {description}. Tried: "
        + ", ".join(candidates)
    )


def safe_float(value):
    try:
        if value is None:
            return np.nan

        number = float(value)

        if np.isfinite(number):
            return number

    except (TypeError, ValueError):
        pass

    return np.nan


def safe_difference(first, second):
    if pd.isna(first) or pd.isna(second):
        return np.nan

    return first - second


def safe_ratio(numerator, denominator):
    if (
        pd.isna(numerator)
        or pd.isna(denominator)
        or denominator == 0
    ):
        return np.nan

    return numerator / denominator


def make_reducer():
    reducer = ee.Reducer.count()

    reducer = reducer.combine(
        reducer2=ee.Reducer.mean(),
        sharedInputs=True,
    )

    reducer = reducer.combine(
        reducer2=ee.Reducer.median(),
        sharedInputs=True,
    )

    reducer = reducer.combine(
        reducer2=ee.Reducer.stdDev(),
        sharedInputs=True,
    )

    reducer = reducer.combine(
        reducer2=ee.Reducer.percentile(
            [10, 25, 75, 90]
        ),
        sharedInputs=True,
    )

    reducer = reducer.combine(
        reducer2=ee.Reducer.minMax(),
        sharedInputs=True,
    )

    return reducer


def extract_band_statistics(
    dictionary,
    band,
):
    return {
        "count": safe_float(
            dictionary.get(
                f"{band}_count"
            )
        ),

        "mean": safe_float(
            dictionary.get(
                f"{band}_mean"
            )
        ),

        "median": safe_float(
            dictionary.get(
                f"{band}_median"
            )
        ),

        "stddev": safe_float(
            dictionary.get(
                f"{band}_stdDev"
            )
        ),

        "p10": safe_float(
            dictionary.get(
                f"{band}_p10"
            )
        ),

        "p25": safe_float(
            dictionary.get(
                f"{band}_p25"
            )
        ),

        "p75": safe_float(
            dictionary.get(
                f"{band}_p75"
            )
        ),

        "p90": safe_float(
            dictionary.get(
                f"{band}_p90"
            )
        ),

        "min": safe_float(
            dictionary.get(
                f"{band}_min"
            )
        ),

        "max": safe_float(
            dictionary.get(
                f"{band}_max"
            )
        ),
    }


def reduce_image_region(
    image,
    geometry,
    reducer,
):
    return (
        image.select(
            [
                CH4_BAND,
                UNCERTAINTY_BAND,
            ]
        )
        .reduceRegion(
            reducer=reducer,
            geometry=geometry,
            scale=GRID_SCALE_M,
            bestEffort=True,
            maxPixels=10000000,
            tileScale=4,
        )
    )


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    project = os.environ.get(
        "EE_PROJECT",
        "methane-release-gee",
    )

    ee.Initialize(project=project)

    frame = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    latitude_column = find_column(
        frame,
        [
            "latitude",
            "s5p_latitude",
            "lat",
        ],
        "latitude",
    )

    longitude_column = find_column(
        frame,
        [
            "longitude",
            "s5p_longitude",
            "lon",
        ],
        "longitude",
    )

    event_time_column = find_column(
        frame,
        [
            "event_time_utc",
            "s5p_event_time_utc",
        ],
        "event time",
    )

    orbit_time_column = find_column(
        frame,
        ["s5p_orbit_time_utc"],
        "S5P orbit time",
    )

    required = [
        "event_id",
        "s5p_system_index",
        latitude_column,
        longitude_column,
        event_time_column,
        orbit_time_column,
    ]

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise KeyError(
            "Missing columns: "
            + ", ".join(missing)
        )

    frame[latitude_column] = pd.to_numeric(
        frame[latitude_column],
        errors="coerce",
    )

    frame[longitude_column] = pd.to_numeric(
        frame[longitude_column],
        errors="coerce",
    )

    frame[event_time_column] = pd.to_datetime(
        frame[event_time_column],
        errors="coerce",
        utc=True,
    )

    frame[orbit_time_column] = pd.to_datetime(
        frame[orbit_time_column],
        errors="coerce",
        utc=True,
    )

    collection = ee.ImageCollection(
        COLLECTION_ID
    )

    reducer = make_reducer()

    records = []

    print("=" * 115)
    print(
        "S5P REGIONAL XCH4 ANOMALY FEATURE EXTRACTION"
    )
    print("=" * 115)

    print("\nInput selected events:", len(frame))
    print("Earth Engine project:", project)

    for number, (_, row) in enumerate(
        frame.iterrows(),
        start=1,
    ):
        event_id = str(row["event_id"])
        image_index = str(
            row["s5p_system_index"]
        )

        latitude = row[latitude_column]
        longitude = row[longitude_column]

        event_time = row[event_time_column]
        orbit_time = row[orbit_time_column]

        print(
            f"\n[{number:03d}/{len(frame):03d}] "
            f"{event_id}"
        )

        record = row.to_dict()

        record.update({
            "s5p_feature_query_success":
                False,

            "s5p_feature_query_error":
                "",

            "source_radius_km":
                SOURCE_RADIUS_M / 1000,

            "background_inner_radius_km":
                BACKGROUND_INNER_RADIUS_M / 1000,

            "background_outer_radius_km":
                BACKGROUND_OUTER_RADIUS_M / 1000,
        })

        if (
            pd.isna(latitude)
            or pd.isna(longitude)
            or pd.isna(event_time)
            or pd.isna(orbit_time)
        ):
            record[
                "s5p_feature_query_error"
            ] = (
                "missing_location_or_time"
            )

            records.append(record)
            continue

        signed_time_difference_hours = (
            (
                orbit_time - event_time
            ).total_seconds()
            / 3600.0
        )

        record.update({
            "signed_orbit_minus_event_hours":
                signed_time_difference_hours,

            "s5p_observation_before_or_after_event":
                (
                    "after_event"
                    if signed_time_difference_hours > 0
                    else (
                        "before_event"
                        if signed_time_difference_hours < 0
                        else "same_time"
                    )
                ),
        })

        try:
            filtered = collection.filter(
                ee.Filter.eq(
                    "system:index",
                    image_index,
                )
            )

            image_count = int(
                filtered.size().getInfo()
            )

            if image_count != 1:
                raise RuntimeError(
                    "Expected one S5P image for "
                    f"{image_index}, found "
                    f"{image_count}"
                )

            image = ee.Image(
                filtered.first()
            )

            point = ee.Geometry.Point(
                [
                    float(longitude),
                    float(latitude),
                ]
            )

            source_region = point.buffer(
                SOURCE_RADIUS_M
            )

            background_outer = point.buffer(
                BACKGROUND_OUTER_RADIUS_M
            )

            background_inner = point.buffer(
                BACKGROUND_INNER_RADIUS_M
            )

            background_region = (
                background_outer.difference(
                    background_inner,
                    maxError=100,
                )
            )

            source_dictionary = (
                reduce_image_region(
                    image,
                    source_region,
                    reducer,
                )
            )

            background_dictionary = (
                reduce_image_region(
                    image,
                    background_region,
                    reducer,
                )
            )

            combined = ee.Dictionary({
                "source":
                    source_dictionary,

                "background":
                    background_dictionary,

                "orbit":
                    image.get("ORBIT"),

                "product_id":
                    image.get("PRODUCT_ID"),

                "product_quality":
                    image.get(
                        "PRODUCT_QUALITY"
                    ),

                "processor_version":
                    image.get(
                        "PROCESSOR_VERSION"
                    ),

                "algorithm_version":
                    image.get(
                        "ALGORITHM_VERSION"
                    ),
            }).getInfo()

            source_raw = (
                combined.get(
                    "source",
                    {},
                )
                or {}
            )

            background_raw = (
                combined.get(
                    "background",
                    {},
                )
                or {}
            )

            source_ch4 = (
                extract_band_statistics(
                    source_raw,
                    CH4_BAND,
                )
            )

            background_ch4 = (
                extract_band_statistics(
                    background_raw,
                    CH4_BAND,
                )
            )

            source_uncertainty = (
                extract_band_statistics(
                    source_raw,
                    UNCERTAINTY_BAND,
                )
            )

            background_uncertainty = (
                extract_band_statistics(
                    background_raw,
                    UNCERTAINTY_BAND,
                )
            )

            mean_anomaly = safe_difference(
                source_ch4["mean"],
                background_ch4["mean"],
            )

            median_anomaly = safe_difference(
                source_ch4["median"],
                background_ch4["median"],
            )

            background_iqr = safe_difference(
                background_ch4["p75"],
                background_ch4["p25"],
            )

            background_robust_sigma = (
                background_iqr / 1.349
                if (
                    pd.notna(background_iqr)
                    and background_iqr > 0
                )
                else np.nan
            )

            mean_z_score = safe_ratio(
                mean_anomaly,
                background_ch4["stddev"],
            )

            median_robust_z_score = safe_ratio(
                median_anomaly,
                background_robust_sigma,
            )

            percent_anomaly = (
                100
                * safe_ratio(
                    mean_anomaly,
                    background_ch4["mean"],
                )
            )

            valid_source = (
                pd.notna(
                    source_ch4["count"]
                )
                and source_ch4["count"] > 0
            )

            valid_background = (
                pd.notna(
                    background_ch4["count"]
                )
                and background_ch4["count"] > 0
            )

            valid_feature = bool(
                valid_source
                and valid_background
                and pd.notna(mean_anomaly)
            )

            record.update({
                "s5p_feature_query_success":
                    True,

                "s5p_feature_valid":
                    valid_feature,

                "confirmed_s5p_orbit":
                    combined.get(
                        "orbit"
                    ),

                "confirmed_s5p_product_id":
                    combined.get(
                        "product_id"
                    ),

                "confirmed_product_quality":
                    combined.get(
                        "product_quality"
                    ),

                "confirmed_processor_version":
                    combined.get(
                        "processor_version"
                    ),

                "confirmed_algorithm_version":
                    combined.get(
                        "algorithm_version"
                    ),

                "source_ch4_gridded_cell_count":
                    source_ch4["count"],

                "source_ch4_mean_ppb":
                    source_ch4["mean"],

                "source_ch4_median_ppb":
                    source_ch4["median"],

                "source_ch4_stddev_ppb":
                    source_ch4["stddev"],

                "source_ch4_p10_ppb":
                    source_ch4["p10"],

                "source_ch4_p25_ppb":
                    source_ch4["p25"],

                "source_ch4_p75_ppb":
                    source_ch4["p75"],

                "source_ch4_p90_ppb":
                    source_ch4["p90"],

                "source_ch4_min_ppb":
                    source_ch4["min"],

                "source_ch4_max_ppb":
                    source_ch4["max"],

                "background_ch4_gridded_cell_count":
                    background_ch4["count"],

                "background_ch4_mean_ppb":
                    background_ch4["mean"],

                "background_ch4_median_ppb":
                    background_ch4["median"],

                "background_ch4_stddev_ppb":
                    background_ch4["stddev"],

                "background_ch4_p10_ppb":
                    background_ch4["p10"],

                "background_ch4_p25_ppb":
                    background_ch4["p25"],

                "background_ch4_p75_ppb":
                    background_ch4["p75"],

                "background_ch4_p90_ppb":
                    background_ch4["p90"],

                "background_ch4_min_ppb":
                    background_ch4["min"],

                "background_ch4_max_ppb":
                    background_ch4["max"],

                "source_uncertainty_mean_ppb":
                    source_uncertainty["mean"],

                "source_uncertainty_median_ppb":
                    source_uncertainty["median"],

                "background_uncertainty_mean_ppb":
                    background_uncertainty["mean"],

                "background_uncertainty_median_ppb":
                    background_uncertainty["median"],

                "source_minus_background_mean_ppb":
                    mean_anomaly,

                "source_minus_background_median_ppb":
                    median_anomaly,

                "source_minus_background_percent":
                    percent_anomaly,

                "background_ch4_iqr_ppb":
                    background_iqr,

                "background_ch4_robust_sigma_ppb":
                    background_robust_sigma,

                "mean_anomaly_z_score":
                    mean_z_score,

                "median_anomaly_robust_z_score":
                    median_robust_z_score,
            })

            print(
                "  source cells:",
                source_ch4["count"],
            )

            print(
                "  background cells:",
                background_ch4["count"],
            )

            print(
                "  mean anomaly ppb:",
                mean_anomaly,
            )

            print(
                "  valid feature:",
                valid_feature,
            )

        except Exception as error:
            print("  ERROR:", error)

            record[
                "s5p_feature_query_error"
            ] = str(error)

            record[
                "s5p_feature_valid"
            ] = False

        records.append(record)

        pd.DataFrame(
            records
        ).to_csv(
            FEATURE_OUTPUT,
            index=False,
        )

        time.sleep(
            REQUEST_DELAY_SECONDS
        )

    result = pd.DataFrame(records)

    result.to_csv(
        FEATURE_OUTPUT,
        index=False,
    )

    valid_mask = (
        result[
            "s5p_feature_query_success"
        ].fillna(False)
        & result[
            "s5p_feature_valid"
        ].fillna(False)
    )

    valid = result[
        valid_mask
    ].copy()

    valid.to_csv(
        VALID_OUTPUT,
        index=False,
    )

    successful_queries = int(
        result[
            "s5p_feature_query_success"
        ].fillna(False).sum()
    )

    primary_valid = valid[
        valid[
            "recommended_analysis_role"
        ].eq(
            "primary_near_time_regional_context"
        )
    ].copy()

    role_summary = (
        result.groupby(
            [
                "recommended_analysis_role",
                "s5p_feature_valid",
            ],
            dropna=False,
        )
        .size()
    )

    if "s5p_true_release" in valid.columns:
        label_summary = (
            valid[
                "s5p_true_release"
            ]
            .value_counts(
                dropna=False
            )
        )
    else:
        label_summary = pd.Series(
            dtype=int
        )

    anomaly_columns = [
        "source_minus_background_mean_ppb",
        "source_minus_background_median_ppb",
        "mean_anomaly_z_score",
        "median_anomaly_robust_z_score",
    ]

    available_anomaly_columns = [
        column
        for column in anomaly_columns
        if column in valid.columns
    ]

    anomaly_summary = (
        valid[
            available_anomaly_columns
        ].describe()
        if (
            not valid.empty
            and available_anomaly_columns
        )
        else pd.DataFrame()
    )

    report_lines = [
        "=" * 115,
        "S5P REGIONAL XCH4 ANOMALY FEATURE REPORT V1",
        "=" * 115,
        "",
        f"Input selected events: {len(result)}",
        (
            "Successful Earth Engine feature queries: "
            f"{successful_queries}"
        ),
        (
            "Events with valid source and background features: "
            f"{len(valid)}"
        ),
        (
            "Primary <=6 h events with valid features: "
            f"{len(primary_valid)}"
        ),
        "",
        "Validity by analysis role:",
        role_summary.to_string(),
        "",
        "Valid feature labels:",
        (
            label_summary.to_string()
            if not label_summary.empty
            else "No label column."
        ),
        "",
        "Anomaly feature statistics:",
        (
            anomaly_summary.to_string()
            if not anomaly_summary.empty
            else "No valid anomaly features."
        ),
        "",
        "Important interpretation:",
        (
            "These features measure regional source-versus-"
            "background XCH4 contrast in a near-time S5P orbit."
        ),
        (
            "They are not confirmed controlled-release plume "
            "measurements because no native pixel acquisition "
            "time was verified inside a release interval."
        ),
        (
            "Gridded cell counts are Earth Engine L3 grid cells "
            "and must not be interpreted as independent native "
            "TROPOMI soundings."
        ),
    ]

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 115)
    print("S5P REGIONAL ANOMALY SUMMARY")
    print("=" * 115)

    print("\nInput selected events:", len(result))

    print(
        "Successful feature queries:",
        successful_queries,
    )

    print(
        "Events with valid anomaly features:",
        len(valid),
    )

    print(
        "Primary <=6 h events with valid features:",
        len(primary_valid),
    )

    print("\nValidity by analysis role:")
    print(role_summary)

    print("\nValid feature labels:")
    print(label_summary)

    print("\nAnomaly feature statistics:")
    print(anomaly_summary)

    print("\nSaved:")
    print(FEATURE_OUTPUT)
    print(VALID_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
