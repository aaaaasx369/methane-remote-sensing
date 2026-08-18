from pathlib import Path
import hashlib
import re

import numpy as np
import pandas as pd
import rasterio


POSITIVE_INPUT = Path(
    "outputs/396_landsat_final_confirmed_features_site_repaired_v1.csv"
)

NEGATIVE_INPUT = Path(
    "outputs/416_landsat_selected_negative_download_manifest_v1.csv"
)

MANIFEST_OUTPUT = Path(
    "outputs/421_landsat_35_scene_benchmark_manifest_v1.csv"
)

QA_OUTPUT = Path(
    "outputs/422_landsat_35_scene_patch_qa_v1.csv"
)

REPORT_OUTPUT = Path(
    "outputs/423_landsat_35_scene_benchmark_report_v1.txt"
)


def find_column(frame, candidates, table_name, required=True):
    for column in candidates:
        if column in frame.columns:
            return column

    if required:
        raise KeyError(
            f"{table_name} 找不到欄位："
            + ", ".join(candidates)
        )

    return None


def normalize_site(value):
    text = str(value).strip().lower()

    if "casa" in text:
        return "casa_grande"

    if "ehrenberg" in text:
        return "ehrenberg"

    if text in {"", "nan", "none", "<na>"}:
        return pd.NA

    return re.sub(
        r"[^a-z0-9]+",
        "_",
        text,
    ).strip("_")


def extract_filename_label(filename):
    match = re.search(
        r"_label_([01])(?:\.tif)?$",
        str(filename),
        flags=re.IGNORECASE,
    )

    if match is None:
        return np.nan

    return int(match.group(1))


def build_positive_manifest():
    frame = pd.read_csv(
        POSITIVE_INPUT,
        low_memory=False,
    )

    frame["label"] = pd.to_numeric(
        frame["label"],
        errors="raise",
    ).astype(int)

    frame = frame[
        frame["label"].eq(1)
    ].copy()

    if len(frame) != 7:
        raise RuntimeError(
            f"預期 7 張 positives，實際為 {len(frame)}。"
        )

    id_column = find_column(
        frame,
        [
            "raster_group_id",
            "sample_id",
            "pixel_hash",
        ],
        "Positive table",
    )

    path_column = find_column(
        frame,
        [
            "relative_path",
            "resolved_patch_path",
            "patch_path",
            "raster_path",
        ],
        "Positive table",
    )

    filename_column = find_column(
        frame,
        [
            "filename",
        ],
        "Positive table",
    )

    site_column = find_column(
        frame,
        [
            "site",
            "site_name_normalized",
            "site_key",
        ],
        "Positive table",
    )

    time_column = find_column(
        frame,
        [
            "landsat_image_time",
            "acquisition_time_utc",
            "datetime_utc",
        ],
        "Positive table",
    )

    sensor_column = find_column(
        frame,
        [
            "landsat_sensor",
            "sensor",
        ],
        "Positive table",
    )

    scene_column = find_column(
        frame,
        [
            "lookup_landsat_product_id",
            "LANDSAT_PRODUCT_ID",
            "landsat_product_id",
            "raster_group_id",
        ],
        "Positive table",
    )

    rate_column = find_column(
        frame,
        [
            "release_rate_kg_h",
            "final_release_rate_kg_h",
            "preferred_release_rate_kg_h",
            "matched_release_rate_kg_h",
            "cr_kgh_CH4_mean300",
        ],
        "Positive table",
        required=False,
    )

    result = pd.DataFrame({
        "sample_id":
            frame[id_column].astype(str),

        "label":
            1,

        "sample_role":
            "confirmed_positive",

        "site":
            frame[site_column].astype(str),

        "site_alias":
            frame[site_column].map(normalize_site),

        "acquisition_time_utc":
            pd.to_datetime(
                frame[time_column],
                errors="coerce",
                utc=True,
            ),

        "landsat_sensor":
            frame[sensor_column].astype(str),

        "scene_id":
            frame[scene_column].astype(str),

        "patch_path":
            frame[path_column].astype(str),

        "source_filename":
            frame[filename_column].astype(str),

        "matched_positive_id":
            frame[id_column].astype(str),

        "pair_slot":
            "positive",

        "temporal_side":
            "positive",

        "days_from_positive":
            0.0,

        "negative_definition":
            pd.NA,

        "label_source":
            "confirmed_release_interval_review",
    })

    if rate_column is None:
        result["release_rate_kg_h"] = np.nan
    else:
        result["release_rate_kg_h"] = pd.to_numeric(
            frame[rate_column],
            errors="coerce",
        )

    result["filename_label"] = (
        result["source_filename"].map(
            extract_filename_label
        )
    )

    result["filename_label_conflict"] = (
        result["filename_label"].notna()
        & result["filename_label"].ne(
            result["label"]
        )
    )

    return result


def build_negative_manifest():
    frame = pd.read_csv(
        NEGATIVE_INPUT,
        low_memory=False,
    )

    if len(frame) != 28:
        raise RuntimeError(
            f"預期 28 張 negatives，實際為 {len(frame)}。"
        )

    site_column = find_column(
        frame,
        [
            "site_alias_standard",
            "positive_site_alias",
            "candidate_site_alias",
            "site_key",
        ],
        "Negative table",
    )

    site_name_column = find_column(
        frame,
        [
            "positive_site",
            "site_name_normalized",
            "site_alias_standard",
        ],
        "Negative table",
    )

    result = pd.DataFrame({
        "sample_id":
            frame["sample_id"].astype(str),

        "label":
            0,

        "sample_role":
            "matched_negative_clean_24h",

        "site":
            frame[site_name_column].astype(str),

        "site_alias":
            frame[site_column].map(normalize_site),

        "acquisition_time_utc":
            pd.to_datetime(
                frame["acquisition_time_utc"],
                errors="coerce",
                utc=True,
            ),

        "landsat_sensor":
            frame[
                "landsat_sensor_standard"
            ].astype(str),

        "scene_id":
            frame[
                "landsat_product_id_standard"
            ].astype(str),

        "patch_path":
            frame["patch_path"].astype(str),

        "source_filename":
            frame["patch_filename"].astype(str),

        "matched_positive_id":
            frame[
                "matched_positive_id"
            ].astype(str),

        "pair_slot":
            frame["pair_slot"].astype(str),

        "temporal_side":
            frame["temporal_side"].astype(str),

        "days_from_positive":
            pd.to_numeric(
                frame["days_from_positive"],
                errors="coerce",
            ),

        "release_rate_kg_h":
            0.0,

        "negative_definition":
            frame[
                "negative_definition"
            ].astype(str),

        "label_source":
            "matched_clean_more_than_24h",
    })

    result["filename_label"] = (
        result["source_filename"].map(
            extract_filename_label
        )
    )

    result["filename_label_conflict"] = (
        result["filename_label"].notna()
        & result["filename_label"].ne(
            result["label"]
        )
    )

    return result


def check_patch(row):
    path = Path(
        str(row["patch_path"])
    )

    record = {
        "sample_id":
            row["sample_id"],

        "label":
            row["label"],

        "sample_role":
            row["sample_role"],

        "site_alias":
            row["site_alias"],

        "patch_path":
            str(path),

        "file_exists":
            path.exists(),

        "file_size_bytes":
            (
                path.stat().st_size
                if path.exists()
                else 0
            ),
    }

    if not path.exists():
        record.update({
            "raster_read_success":
                False,

            "raster_error":
                "file_not_found",
        })

        return record

    try:
        with rasterio.open(path) as source:
            array = source.read()

            pixel_hash = hashlib.sha256(
                array.tobytes()
            ).hexdigest()

            record.update({
                "raster_read_success":
                    True,

                "band_count":
                    source.count,

                "height":
                    source.height,

                "width":
                    source.width,

                "dtype":
                    str(source.dtypes[0]),

                "crs":
                    str(source.crs),

                "pixel_size_x":
                    abs(
                        float(
                            source.transform.a
                        )
                    ),

                "pixel_size_y":
                    abs(
                        float(
                            source.transform.e
                        )
                    ),

                "all_zero":
                    bool(
                        np.all(array == 0)
                    ),

                "has_nan":
                    bool(
                        np.isnan(
                            array.astype(
                                "float64"
                            )
                        ).any()
                    ),

                "finite_fraction":
                    float(
                        np.isfinite(
                            array
                        ).mean()
                    ),

                "zero_fraction":
                    float(
                        (
                            array == 0
                        ).mean()
                    ),

                "pixel_min":
                    float(
                        np.nanmin(array)
                    ),

                "pixel_max":
                    float(
                        np.nanmax(array)
                    ),

                "pixel_mean":
                    float(
                        np.nanmean(array)
                    ),

                "pixel_hash":
                    pixel_hash,
            })

    except Exception as error:
        record.update({
            "raster_read_success":
                False,

            "raster_error":
                str(error),
        })

    return record


def main():
    positives = build_positive_manifest()
    negatives = build_negative_manifest()

    manifest = pd.concat(
        [
            positives,
            negatives,
        ],
        ignore_index=True,
        sort=False,
    )

    manifest[
        "benchmark_name"
    ] = "landsat_matched_controlled_release_v1"

    manifest[
        "benchmark_design"
    ] = "2_before_2_after_global_unique_clean_24h"

    manifest = manifest.sort_values(
        [
            "site_alias",
            "matched_positive_id",
            "label",
            "acquisition_time_utc",
        ],
        ascending=[
            True,
            True,
            False,
            True,
        ],
    ).reset_index(drop=True)

    if len(manifest) != 35:
        raise RuntimeError(
            f"應有 35 scenes，實際為 {len(manifest)}。"
        )

    if manifest["sample_id"].duplicated().any():
        raise RuntimeError(
            "sample_id 有重複。"
        )

    if manifest["patch_path"].duplicated().any():
        raise RuntimeError(
            "patch_path 有重複。"
        )

    label_counts = (
        manifest["label"]
        .value_counts()
        .sort_index()
    )

    if int(label_counts.get(0, 0)) != 28:
        raise RuntimeError(
            "Negative count 不是 28。"
        )

    if int(label_counts.get(1, 0)) != 7:
        raise RuntimeError(
            "Positive count 不是 7。"
        )

    negative = manifest[
        manifest["label"].eq(0)
    ]

    allocation = pd.crosstab(
        negative["matched_positive_id"],
        negative["temporal_side"],
    )

    for side in ["before", "after"]:
        if side not in allocation.columns:
            raise RuntimeError(
                f"缺少 {side} 配對。"
            )

        if not allocation[side].eq(2).all():
            raise RuntimeError(
                f"並非每個 positive 都有 2 個 {side}。"
            )

    qa = pd.DataFrame(
        [
            check_patch(row)
            for _, row in manifest.iterrows()
        ]
    )

    qa["qa_pass"] = (
        qa["file_exists"].eq(True)
        & qa["raster_read_success"].eq(True)
        & qa["band_count"].eq(6)
        & qa["all_zero"].eq(False)
        & qa["has_nan"].eq(False)
    )

    duplicate_hash_mask = (
        qa["pixel_hash"]
        .notna()
        & qa["pixel_hash"].duplicated(
            keep=False
        )
    )

    qa[
        "duplicate_pixel_raster"
    ] = duplicate_hash_mask

    qa.to_csv(
        QA_OUTPUT,
        index=False,
    )

    manifest = manifest.merge(
        qa[
            [
                "sample_id",
                "qa_pass",
                "band_count",
                "height",
                "width",
                "crs",
                "pixel_hash",
                "duplicate_pixel_raster",
            ]
        ],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )

    manifest[
        "benchmark_ready"
    ] = (
        manifest["qa_pass"].eq(True)
        & manifest[
            "duplicate_pixel_raster"
        ].eq(False)
    )

    manifest.to_csv(
        MANIFEST_OUTPUT,
        index=False,
    )

    label_by_site = pd.crosstab(
        manifest["site_alias"],
        manifest["label"],
        margins=True,
    )

    label_by_sensor = pd.crosstab(
        manifest["landsat_sensor"],
        manifest["label"],
        margins=True,
    )

    shape_counts = (
        qa.groupby(
            [
                "height",
                "width",
            ]
        )
        .size()
        .sort_values(
            ascending=False
        )
    )

    filename_conflicts = manifest[
        manifest[
            "filename_label_conflict"
        ].eq(True)
    ]

    report_lines = [
        "=" * 110,
        "LANDSAT 35-SCENE MATCHED BENCHMARK REPORT V1",
        "=" * 110,
        "",
        f"Total scenes: {len(manifest)}",
        f"Positive scenes: {int(manifest['label'].eq(1).sum())}",
        f"Negative scenes: {int(manifest['label'].eq(0).sum())}",
        (
            "QA-pass scenes: "
            f"{int(manifest['qa_pass'].sum())}"
        ),
        (
            "Benchmark-ready scenes: "
            f"{int(manifest['benchmark_ready'].sum())}"
        ),
        (
            "Duplicated pixel rasters: "
            f"{int(manifest['duplicate_pixel_raster'].sum())}"
        ),
        "",
        "Label by site:",
        label_by_site.to_string(),
        "",
        "Label by Landsat sensor:",
        label_by_sensor.to_string(),
        "",
        "Before/after allocation:",
        allocation.to_string(),
        "",
        "Raster shapes:",
        shape_counts.to_string(),
        "",
        (
            "Filename/final-label conflicts: "
            f"{len(filename_conflicts)}"
        ),
        (
            filename_conflicts[
                [
                    "sample_id",
                    "source_filename",
                    "filename_label",
                    "label",
                ]
            ].to_string(index=False)
            if not filename_conflicts.empty
            else "None"
        ),
        "",
        (
            "Important: final benchmark labels come from "
            "scene-level release interval review and the "
            "locked matched-negative manifest, not filenames."
        ),
    ]

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("=" * 110)
    print(
        "LANDSAT 35-SCENE BENCHMARK COMPLETE"
    )
    print("=" * 110)

    print("\nTotal scenes:", len(manifest))
    print("Positive scenes:", int(manifest["label"].eq(1).sum()))
    print("Negative scenes:", int(manifest["label"].eq(0).sum()))
    print("QA-pass scenes:", int(manifest["qa_pass"].sum()))
    print(
        "Benchmark-ready scenes:",
        int(manifest["benchmark_ready"].sum()),
    )

    print("\nLabel by site:")
    print(label_by_site)

    print("\nLabel by sensor:")
    print(label_by_sensor)

    print("\nRaster shapes:")
    print(shape_counts)

    print(
        "\nFilename/final-label conflicts:",
        len(filename_conflicts),
    )

    if not filename_conflicts.empty:
        print(
            filename_conflicts[
                [
                    "sample_id",
                    "source_filename",
                    "filename_label",
                    "label",
                ]
            ].to_string(index=False)
        )

    print(
        "\nDuplicated pixel rasters:",
        int(
            manifest[
                "duplicate_pixel_raster"
            ].sum()
        ),
    )

    print("\nSaved:")
    print(MANIFEST_OUTPUT)
    print(QA_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
