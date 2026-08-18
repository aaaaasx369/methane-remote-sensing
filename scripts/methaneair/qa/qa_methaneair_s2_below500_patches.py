from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform


INPUT = Path(
    "outputs/455_methaneair_s2_below500_patch_index_v1.csv"
)

QA_OUTPUT = Path(
    "outputs/456_methaneair_s2_below500_patch_qa_v1.csv"
)

CANDIDATE_OUTPUT = Path(
    "outputs/457_methaneair_s2_below500_candidate_manifest_v1.csv"
)

REPORT_OUTPUT = Path(
    "outputs/458_methaneair_s2_below500_qa_report_v1.txt"
)


EXPECTED_ROWS = 21
EXPECTED_BANDS = 6
EXPECTED_WIDTH = 101
EXPECTED_HEIGHT = 101
EXPECTED_PIXEL_SIZE_M = 20.0


def find_column(frame, candidates, description):
    for column in candidates:
        if column in frame.columns:
            return column

    raise KeyError(
        f"Cannot find {description}. Tried: "
        + ", ".join(candidates)
    )


def classify_time_gap(hours):
    if pd.isna(hours):
        return "unknown"

    if hours <= 1:
        return "tier_A_within_1h"

    if hours <= 3:
        return "tier_B_1_to_3h"

    if hours <= 6:
        return "tier_C_3_to_6h"

    return "outside_6h"


def classify_emission(rate):
    if pd.isna(rate):
        return "unknown"

    if rate < 200:
        return "0_to_200"

    if rate < 500:
        return "200_to_500"

    return "outside_below500"


def inspect_raster(row):
    path = Path(str(row["patch_path"]))

    record = {
        "event_id": row["event_id"],
        "scene_id": row["scene_id"],
        "patch_path": str(path),
        "emission_kg_hr": row["emission_kg_hr"],
        "absolute_time_difference_hours":
            row["absolute_time_difference_hours"],
        "time_match_tier": row["time_match_tier"],
        "file_exists": path.exists(),
        "file_size_bytes":
            path.stat().st_size if path.exists() else 0,
    }

    if not path.exists():
        record.update({
            "raster_read_success": False,
            "raster_error": "file_not_found",
        })
        return record

    try:
        with rasterio.open(path) as source:
            array = source.read()

            longitude = float(row["longitude"])
            latitude = float(row["latitude"])

            projected_x, projected_y = transform(
                "EPSG:4326",
                source.crs,
                [longitude],
                [latitude],
            )

            source_inside_bounds = (
                source.bounds.left
                <= projected_x[0]
                <= source.bounds.right
                and source.bounds.bottom
                <= projected_y[0]
                <= source.bounds.top
            )

            pixel_size_x = abs(float(source.transform.a))
            pixel_size_y = abs(float(source.transform.e))

            record.update({
                "raster_read_success": True,
                "band_count": source.count,
                "width": source.width,
                "height": source.height,
                "dtype": str(source.dtypes[0]),
                "crs": str(source.crs),
                "pixel_size_x": pixel_size_x,
                "pixel_size_y": pixel_size_y,
                "all_zero": bool(np.all(array == 0)),
                "has_nan": bool(
                    np.isnan(
                        array.astype("float64")
                    ).any()
                ),
                "zero_fraction": float(
                    np.mean(array == 0)
                ),
                "minimum": float(np.nanmin(array)),
                "maximum": float(np.nanmax(array)),
                "mean": float(np.nanmean(array)),
                "source_inside_bounds":
                    bool(source_inside_bounds),
                "band_count_pass":
                    source.count == EXPECTED_BANDS,
                # Earth Engine aligns exports to each Sentinel-2
                # scene's native projected pixel grid. A nominal
                # 2 km region at 20 m can therefore be 100–104 pixels.
                "shape_pass":
                    (
                        100 <= source.width <= 104
                        and 100 <= source.height <= 104
                    ),

                "physical_width_m":
                    float(source.width * pixel_size_x),

                "physical_height_m":
                    float(source.height * pixel_size_y),

                "physical_extent_pass":
                    (
                        2000 <= source.width * pixel_size_x <= 2080
                        and 2000 <= source.height * pixel_size_y <= 2080
                    ),
                "dtype_pass":
                    all(
                        dtype == "uint16"
                        for dtype in source.dtypes
                    ),
                "pixel_size_pass":
                    (
                        abs(
                            pixel_size_x
                            - EXPECTED_PIXEL_SIZE_M
                        ) < 1e-6
                        and abs(
                            pixel_size_y
                            - EXPECTED_PIXEL_SIZE_M
                        ) < 1e-6
                    ),
            })

    except Exception as error:
        record.update({
            "raster_read_success": False,
            "raster_error": str(error),
        })

    return record


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    frame = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    if len(frame) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} rows, "
            f"found {len(frame)}."
        )

    event_column = find_column(
        frame,
        ["event_id"],
        "event ID",
    )

    scene_column = find_column(
        frame,
        ["scene_id"],
        "scene ID",
    )

    patch_column = find_column(
        frame,
        ["patch_path", "relative_path"],
        "patch path",
    )

    latitude_column = find_column(
        frame,
        ["latitude", "lat"],
        "latitude",
    )

    longitude_column = find_column(
        frame,
        ["longitude", "lon"],
        "longitude",
    )

    emission_column = find_column(
        frame,
        ["emission_kg_hr", "release_rate_kg_h"],
        "emission rate",
    )

    event_time_column = find_column(
        frame,
        ["event_time_utc", "datetime_utc"],
        "MethaneAIR event time",
    )

    acquisition_column = find_column(
        frame,
        [
            "actual_acquisition_time_utc",
            "acquisition_time_utc",
        ],
        "Sentinel-2 acquisition time",
    )

    frame["event_id"] = frame[event_column].astype(str)
    frame["scene_id"] = frame[scene_column].astype(str)
    frame["patch_path"] = frame[patch_column].astype(str)

    frame["latitude"] = pd.to_numeric(
        frame[latitude_column],
        errors="coerce",
    )

    frame["longitude"] = pd.to_numeric(
        frame[longitude_column],
        errors="coerce",
    )

    frame["emission_kg_hr"] = pd.to_numeric(
        frame[emission_column],
        errors="coerce",
    )

    frame["event_time_utc_clean"] = pd.to_datetime(
        frame[event_time_column],
        errors="coerce",
        utc=True,
    )

    frame[
        "s2_acquisition_time_utc_clean"
    ] = pd.to_datetime(
        frame[acquisition_column],
        errors="coerce",
        utc=True,
    )

    frame[
        "absolute_time_difference_hours"
    ] = (
        frame["s2_acquisition_time_utc_clean"]
        - frame["event_time_utc_clean"]
    ).abs().dt.total_seconds() / 3600.0

    frame["time_match_tier"] = frame[
        "absolute_time_difference_hours"
    ].map(classify_time_gap)

    frame["emission_bin"] = frame[
        "emission_kg_hr"
    ].map(classify_emission)

    if frame["event_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate event_id values found."
        )

    qa = pd.DataFrame(
        [
            inspect_raster(row)
            for _, row in frame.iterrows()
        ]
    )

    # This column only exists when a raster read actually fails.
    # Create it for rows that failed another QA condition.
    if "raster_error" not in qa.columns:
        qa["raster_error"] = ""

    required_boolean_columns = [
        "file_exists",
        "raster_read_success",
        "band_count_pass",
        "shape_pass",
        "dtype_pass",
        "pixel_size_pass",
        "source_inside_bounds",
    ]

    for column in required_boolean_columns:
        if column not in qa.columns:
            qa[column] = False

    qa["qa_pass"] = (
        qa["file_exists"].eq(True)
        & qa["raster_read_success"].eq(True)
        & qa["band_count_pass"].eq(True)
        & qa["shape_pass"].eq(True)
        & qa["physical_extent_pass"].eq(True)
        & qa["dtype_pass"].eq(True)
        & qa["pixel_size_pass"].eq(True)
        & qa["source_inside_bounds"].eq(True)
        & qa["all_zero"].eq(False)
        & qa["has_nan"].eq(False)
    )

    qa.to_csv(
        QA_OUTPUT,
        index=False,
    )

    candidate = frame.merge(
        qa[
            [
                "event_id",
                "qa_pass",
                "band_count",
                "width",
                "height",
                "dtype",
                "crs",
                "zero_fraction",
                "source_inside_bounds",
            ]
        ],
        on="event_id",
        how="left",
        validate="one_to_one",
    )

    candidate["candidate_status"] = np.where(
        candidate["qa_pass"],
        "external_low_emission_candidate",
        "exclude_failed_patch_qa",
    )

    candidate[
        "positive_label_status"
    ] = "not_yet_locked_due_to_temporal_gap"

    candidate[
        "evaluation_group"
    ] = candidate["scene_id"]

    candidate.to_csv(
        CANDIDATE_OUTPUT,
        index=False,
    )

    time_summary = (
        candidate["time_match_tier"]
        .value_counts()
        .reindex(
            [
                "tier_A_within_1h",
                "tier_B_1_to_3h",
                "tier_C_3_to_6h",
                "outside_6h",
                "unknown",
            ],
            fill_value=0,
        )
    )

    emission_summary = (
        candidate["emission_bin"]
        .value_counts()
        .reindex(
            [
                "0_to_200",
                "200_to_500",
                "outside_below500",
                "unknown",
            ],
            fill_value=0,
        )
    )

    shape_summary = (
        qa.groupby(
            ["height", "width"],
            dropna=False,
        )
        .size()
        .sort_values(ascending=False)
    )

    report_lines = [
        "=" * 105,
        "METHANEAIR–S2 BELOW-500 KG/H PATCH QA V1",
        "=" * 105,
        "",
        f"Expected patches: {EXPECTED_ROWS}",
        f"Files found: {int(qa['file_exists'].sum())}",
        (
            "Readable rasters: "
            f"{int(qa['raster_read_success'].sum())}"
        ),
        f"QA-pass patches: {int(qa['qa_pass'].sum())}",
        (
            "Unique MethaneAIR events: "
            f"{candidate['event_id'].nunique()}"
        ),
        (
            "Unique Sentinel-2 scenes: "
            f"{candidate['scene_id'].nunique()}"
        ),
        "",
        "Temporal match tiers:",
        time_summary.to_string(),
        "",
        "Emission bins:",
        emission_summary.to_string(),
        "",
        "Raster shapes:",
        shape_summary.to_string(),
        "",
        (
            "Important: these scenes remain external "
            "low-emission candidates. They are not yet "
            "locked positive labels because MethaneAIR and "
            "Sentinel-2 were not acquired simultaneously."
        ),
    ]

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("=" * 105)
    print(
        "METHANEAIR–S2 BELOW-500 KG/H QA"
    )
    print("=" * 105)

    print(
        "\nExpected patches:",
        EXPECTED_ROWS,
    )

    print(
        "Files found:",
        int(qa["file_exists"].sum()),
    )

    print(
        "Readable rasters:",
        int(
            qa["raster_read_success"].sum()
        ),
    )

    print(
        "QA-pass patches:",
        int(qa["qa_pass"].sum()),
    )

    print(
        "Unique events:",
        candidate["event_id"].nunique(),
    )

    print(
        "Unique Sentinel-2 scenes:",
        candidate["scene_id"].nunique(),
    )

    print("\nTemporal match tiers:")
    print(time_summary)

    print("\nEmission bins:")
    print(emission_summary)

    print("\nRaster shapes:")
    print(shape_summary)

    failed = qa[
        ~qa["qa_pass"]
    ]

    if not failed.empty:
        print("\nFailed QA:")
        print(
            failed[
                [
                    "event_id",
                    "file_exists",
                    "raster_read_success",
                    "band_count",
                    "width",
                    "height",
                    "dtype",
                    "pixel_size_x",
                    "pixel_size_y",
                    "all_zero",
                    "has_nan",
                    "source_inside_bounds",
                    "raster_error",
                ]
            ].to_string(index=False)
        )

    print("\nSaved:")
    print(QA_OUTPUT)
    print(CANDIDATE_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
