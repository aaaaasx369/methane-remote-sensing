from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


MANIFEST_INPUT = Path(
    "outputs/416_landsat_selected_negative_download_manifest_v1.csv"
)

DOWNLOAD_INDEX_INPUT = Path(
    "outputs/417_landsat_selected_negative_download_index_v1.csv"
)

QA_OUTPUT = Path(
    "outputs/418_landsat_matched_negative_patch_qa_v1.csv"
)

REPORT_OUTPUT = Path(
    "outputs/419_landsat_matched_negative_patch_qa_report_v1.txt"
)


def main():
    manifest = pd.read_csv(
        MANIFEST_INPUT,
        low_memory=False,
    )

    records = []

    for _, row in manifest.iterrows():
        patch_path = Path(
            str(row["patch_path"])
        )

        record = {
            "sample_id":
                row["sample_id"],

            "patch_path":
                str(patch_path),

            "file_exists":
                patch_path.exists(),

            "file_size_bytes":
                (
                    patch_path.stat().st_size
                    if patch_path.exists()
                    else 0
                ),
        }

        if not patch_path.exists():
            record.update({
                "raster_read_success":
                    False,

                "raster_error":
                    "file_not_found",
            })

            records.append(record)
            continue

        try:
            with rasterio.open(
                patch_path
            ) as source:
                array = source.read()

                finite = np.isfinite(
                    array
                )

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
                        str(
                            source.dtypes[0]
                        ),

                    "crs":
                        str(source.crs),

                    "transform":
                        str(
                            source.transform
                        ),

                    "all_zero":
                        bool(
                            np.all(
                                array == 0
                            )
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
                            finite.mean()
                        ),

                    "zero_fraction":
                        float(
                            (
                                array == 0
                            ).mean()
                        ),

                    "pixel_min":
                        float(
                            np.nanmin(
                                array
                            )
                        ),

                    "pixel_max":
                        float(
                            np.nanmax(
                                array
                            )
                        ),

                    "pixel_mean":
                        float(
                            np.nanmean(
                                array
                            )
                        ),
                })

        except Exception as error:
            record.update({
                "raster_read_success":
                    False,

                "raster_error":
                    str(error),
            })

        records.append(record)

    qa = pd.DataFrame(records)

    qa[
        "qa_pass"
    ] = (
        qa[
            "file_exists"
        ].eq(True)
        & qa[
            "raster_read_success"
        ].eq(True)
        & qa[
            "band_count"
        ].eq(6)
        & qa[
            "all_zero"
        ].eq(False)
        & qa[
            "has_nan"
        ].eq(False)
    )

    qa.to_csv(
        QA_OUTPUT,
        index=False,
    )

    status_counts = (
        qa["qa_pass"]
        .value_counts(
            dropna=False
        )
    )

    band_counts = (
        qa["band_count"]
        .value_counts(
            dropna=False
        )
    )

    shape_counts = (
        qa.groupby(
            [
                "height",
                "width",
            ],
            dropna=False,
        )
        .size()
        .sort_values(
            ascending=False
        )
    )

    report_lines = [
        "=" * 100,
        "LANDSAT MATCHED-NEGATIVE PATCH QA V1",
        "=" * 100,
        "",
        f"Expected patches: {len(manifest)}",
        (
            "Files found: "
            f"{int(qa['file_exists'].sum())}"
        ),
        (
            "Readable rasters: "
            f"{int(qa['raster_read_success'].sum())}"
        ),
        (
            "QA-pass patches: "
            f"{int(qa['qa_pass'].sum())}"
        ),
        "",
        "QA status:",
        status_counts.to_string(),
        "",
        "Band counts:",
        band_counts.to_string(),
        "",
        "Raster shapes:",
        shape_counts.to_string(),
        "",
        (
            "All-zero patches: "
            f"{int(qa['all_zero'].fillna(False).sum())}"
        ),
        (
            "Patches containing NaN: "
            f"{int(qa['has_nan'].fillna(False).sum())}"
        ),
    ]

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("=" * 100)
    print(
        "LANDSAT MATCHED-NEGATIVE PATCH QA"
    )
    print("=" * 100)

    print(
        "\nExpected patches:",
        len(manifest),
    )

    print(
        "Files found:",
        int(
            qa[
                "file_exists"
            ].sum()
        ),
    )

    print(
        "Readable rasters:",
        int(
            qa[
                "raster_read_success"
            ].sum()
        ),
    )

    print(
        "QA-pass patches:",
        int(
            qa[
                "qa_pass"
            ].sum()
        ),
    )

    print("\nQA status:")
    print(status_counts)

    print("\nBand counts:")
    print(band_counts)

    print("\nRaster shapes:")
    print(shape_counts)

    failed = qa[
        ~qa["qa_pass"]
    ]

    if not failed.empty:
        print("\nFailed QA:")
        print(
            failed[
                [
                    "sample_id",
                    "file_exists",
                    "raster_read_success",
                    "band_count",
                    "all_zero",
                    "has_nan",
                    "raster_error",
                ]
            ].to_string(
                index=False
            )
        )

    print("\nSaved:")
    print(QA_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
