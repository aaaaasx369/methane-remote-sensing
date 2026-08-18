from pathlib import Path
import re

import pandas as pd

import download_and_qa_s2_low_emission_scenes as base


MANIFEST_INPUT = Path(
    "outputs/324_s2_low_emission_matched_negative_manifest_v2.csv"
)

PATCH_DIR = Path(
    "sample_patches/s2_low_emission_negatives_v2"
)

PREVIEW_DIR = Path(
    "outputs/s2_low_emission_negative_previews_v2"
)

INDEX_OUTPUT = Path(
    "outputs/325_s2_low_emission_negative_patch_index_v2.csv"
)

QA_OUTPUT = Path(
    "outputs/326_s2_low_emission_negative_local_qa_v2.csv"
)


def safe_name(text):
    return re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        str(text),
    ).strip("_")


def main():
    if not MANIFEST_INPUT.exists():
        raise FileNotFoundError(
            MANIFEST_INPUT
        )

    base.initialize_earth_engine()

    PATCH_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREVIEW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 原本 QA 函式會使用模組中的 PREVIEW_DIR。
    base.PREVIEW_DIR = PREVIEW_DIR

    manifest = pd.read_csv(
        MANIFEST_INPUT,
        low_memory=False,
    )

    required_columns = [
        "negative_id",
        "site",
        "scene_id",
        "acquisition_time_utc",
        "lat",
        "lon",
        "matched_positive_scene_id",
        "matched_positive_time_utc",
        "matched_positive_rate_kg_h",
        "label",
    ]

    missing = [
        column
        for column in required_columns
        if column not in manifest.columns
    ]

    if missing:
        raise KeyError(
            "Negative manifest 缺少欄位："
            + ", ".join(missing)
        )

    manifest["lat"] = pd.to_numeric(
        manifest["lat"],
        errors="coerce",
    )

    manifest["lon"] = pd.to_numeric(
        manifest["lon"],
        errors="coerce",
    )

    manifest[
        "acquisition_time_utc"
    ] = pd.to_datetime(
        manifest[
            "acquisition_time_utc"
        ],
        errors="coerce",
        utc=True,
    )

    manifest[
        "matched_positive_time_utc"
    ] = pd.to_datetime(
        manifest[
            "matched_positive_time_utc"
        ],
        errors="coerce",
        utc=True,
    )

    manifest = manifest.dropna(
        subset=[
            "negative_id",
            "scene_id",
            "lat",
            "lon",
            "acquisition_time_utc",
        ]
    ).copy()

    index_rows = []
    qa_rows = []

    print("=" * 110)
    print(
        "DOWNLOAD AND QA SENTINEL-2 "
        "LOW-EMISSION MATCHED NEGATIVES"
    )
    print("=" * 110)

    print(
        "\nNegative scenes:",
        len(manifest),
    )

    for number, row in (
        manifest.reset_index(
            drop=True
        ).iterrows()
    ):
        acquisition = row[
            "acquisition_time_utc"
        ]

        date_text = acquisition.strftime(
            "%Y%m%dT%H%M%S"
        )

        stem = safe_name(
            f"{row['negative_id']}_"
            f"{date_text}_"
            f"{row['site']}"
        )

        tif_path = PATCH_DIR / (
            f"{stem}.tif"
        )

        print(
            f"\n[{number + 1}/{len(manifest)}] "
            f"{row['negative_id']} | "
            f"{row['site']} | "
            f"{acquisition}",
            flush=True,
        )

        print(
            "  Scene:",
            row["scene_id"],
        )

        if tif_path.exists():
            download_status = "existing"

            print(
                "  Existing patch:",
                tif_path,
            )

        else:
            try:
                base.download_scene(
                    scene_id=
                        row["scene_id"],
                    latitude=
                        float(row["lat"]),
                    longitude=
                        float(row["lon"]),
                    output_path=
                        tif_path,
                )

                download_status = "downloaded"

                print(
                    "  Downloaded:",
                    tif_path,
                )

            except Exception as error:
                print(
                    "  Download failed:",
                    error,
                )

                index_rows.append({
                    **row.to_dict(),

                    "patch_path":
                        str(tif_path),

                    "download_status":
                        "failed",

                    "download_error":
                        str(error),
                })

                continue

        index_record = {
            **row.to_dict(),

            "patch_path":
                str(tif_path),

            "download_status":
                download_status,

            "download_error":
                "",
        }

        index_rows.append(
            index_record
        )

        try:
            qa = base.run_local_qa(
                tif_path=tif_path,
                latitude=float(
                    row["lat"]
                ),
                longitude=float(
                    row["lon"]
                ),
                preview_stem=stem,
            )

            qa_record = {
                **index_record,
                **qa,
            }

            qa_rows.append(
                qa_record
            )

            print(
                "  Local valid:",
                f"{qa['local_valid_fraction']:.3f}",
            )

            print(
                "  Local cloud/shadow/snow:",
                f"{qa['local_bad_atmosphere_fraction']:.3f}",
            )

            print(
                "  Preliminary QA pass:",
                qa[
                    "qa_pass_preliminary"
                ],
            )

        except Exception as error:
            print(
                "  QA failed:",
                error,
            )

            qa_rows.append({
                **index_record,

                "qa_pass_preliminary":
                    False,

                "qa_error":
                    str(error),
            })

    index = pd.DataFrame(
        index_rows
    )

    qa_table = pd.DataFrame(
        qa_rows
    )

    index.to_csv(
        INDEX_OUTPUT,
        index=False,
    )

    qa_table.to_csv(
        QA_OUTPUT,
        index=False,
    )

    print("\n" + "=" * 110)
    print(
        "NEGATIVE PATCH QA SUMMARY"
    )
    print("=" * 110)

    print(
        "\nManifest negatives:",
        len(manifest),
    )

    downloaded_count = (
        int(
            index[
                "download_status"
            ].isin([
                "downloaded",
                "existing",
            ]).sum()
        )
        if not index.empty
        else 0
    )

    print(
        "Downloaded/existing patches:",
        downloaded_count,
    )

    if not qa_table.empty:
        pass_flag = (
            qa_table[
                "qa_pass_preliminary"
            ]
            .astype(str)
            .str.lower()
            .isin([
                "true",
                "1",
                "yes",
            ])
        )

        print(
            "Preliminary QA passes:",
            int(pass_flag.sum()),
        )

        print(
            "Preliminary QA failures:",
            int((~pass_flag).sum()),
        )

        print("\nLocal QA statistics:")

        for column in [
            "local_valid_fraction",
            "local_cloud_fraction",
            "local_shadow_fraction",
            "local_bad_atmosphere_fraction",
            "local_all_zero_fraction",
        ]:
            if column in qa_table.columns:
                print(f"\n{column}:")
                print(
                    pd.to_numeric(
                        qa_table[column],
                        errors="coerce",
                    ).describe()
                )

        display_columns = [
            "negative_id",
            "matched_positive_rate_kg_h",
            "acquisition_time_utc",
            "days_from_positive",
            "temporal_side",
            "scene_cloud_percentage",
            "nearest_nonzero_release_hours",
            "source_center_inside_raster",
            "local_valid_fraction",
            "local_cloud_fraction",
            "local_shadow_fraction",
            "local_bad_atmosphere_fraction",
            "local_all_zero_fraction",
            "qa_pass_preliminary",
            "patch_path",
        ]

        display_columns = [
            column
            for column in display_columns
            if column in qa_table.columns
        ]

        print("\nNegative QA table:")

        print(
            qa_table[
                display_columns
            ].to_string(
                index=False,
            )
        )

    print("\nSaved:")
    print(INDEX_OUTPUT)
    print(QA_OUTPUT)
    print(PREVIEW_DIR)


if __name__ == "__main__":
    main()
