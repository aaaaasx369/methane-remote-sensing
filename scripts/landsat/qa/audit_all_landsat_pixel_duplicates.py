from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import rasterio


INPUT_TABLES = {
    "existing_extended_dataset": Path(
        "outputs/80_landsat_extended_schedule_confirmed_features.csv"
    ),
    "targeted_casa_grande": Path(
        "outputs/82_landsat_targeted_reviewed_features.csv"
    ),
    "ehrenberg_priority": Path(
        "outputs/90_ehrenberg_priority_landsat_patch_index.csv"
    ),
}

AUDIT_OUTPUT = Path(
    "outputs/91_all_landsat_cross_dataset_hash_audit.csv"
)

DUPLICATE_OUTPUT = Path(
    "outputs/92_landsat_cross_dataset_duplicate_groups.csv"
)

PATH_COLUMNS = [
    "resolved_patch_path",
    "file_path",
    "patch_path",
    "tif_path",
    "filepath",
    "image_path",
    "output_path",
    "filename",
    "patch_filename",
]

SEARCH_ROOTS = [
    Path("sample_patches"),
    Path("outputs"),
    Path("data"),
    Path("downloads"),
]


def clean_value(value):
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.lower() in {
        "",
        "nan",
        "none",
        "null",
    }:
        return ""

    return text


def build_filename_index():
    index = {}

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        for pattern in ("*.tif", "*.tiff"):
            for path in root.rglob(pattern):
                index.setdefault(
                    path.name,
                    [],
                ).append(path.resolve())

    return index


def resolve_patch_path(row, filename_index):
    attempted_values = []

    for column in PATH_COLUMNS:
        if column not in row.index:
            continue

        value = clean_value(row[column])

        if not value:
            continue

        attempted_values.append(value)

        candidate = Path(value).expanduser()

        if candidate.exists():
            return candidate.resolve()

        project_candidate = (
            Path.cwd() / candidate
        )

        if project_candidate.exists():
            return project_candidate.resolve()

        matches = filename_index.get(
            candidate.name,
            [],
        )

        if len(matches) == 1:
            return matches[0]

        if len(matches) > 1:
            exact_suffix_matches = [
                match
                for match in matches
                if str(match).endswith(value)
            ]

            if len(exact_suffix_matches) == 1:
                return exact_suffix_matches[0]

            raise RuntimeError(
                "Multiple TIFF files match "
                f"{candidate.name}:\n"
                + "\n".join(
                    str(match)
                    for match in matches
                )
            )

    raise FileNotFoundError(
        "Could not resolve TIFF path. "
        f"Attempted values: {attempted_values}"
    )


def calculate_canonical_pixel_hash(array):
    """
    Hash includes array shape, dtype and pixel values.

    Including shape prevents two arrays with the same byte
    sequence but different dimensions from being considered equal.
    """
    contiguous = np.ascontiguousarray(array)

    digest = hashlib.sha256()

    digest.update(
        str(contiguous.shape).encode("utf-8")
    )

    digest.update(
        str(contiguous.dtype).encode("utf-8")
    )

    digest.update(
        contiguous.tobytes()
    )

    return digest.hexdigest()


def first_available(row, candidates):
    for column in candidates:
        if column not in row.index:
            continue

        value = row[column]

        if pd.notna(value):
            return value

    return np.nan


def load_table(source_name, path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input table: {path}"
        )

    dataframe = pd.read_csv(
        path,
        low_memory=False,
    )

    if "download_status" in dataframe.columns:
        successful_statuses = {
            "success",
            "success_existing",
        }

        status_text = (
            dataframe["download_status"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        successful_mask = status_text.isin(
            successful_statuses
        )

        # Feature tables might not use download_status.
        # Apply the filter only when at least one row matches.
        if successful_mask.any():
            dataframe = dataframe[
                successful_mask
            ].copy()

    dataframe["dataset_source"] = (
        source_name
    )

    return dataframe


def main():
    print("=" * 100)
    print("ALL LANDSAT CROSS-DATASET PIXEL-HASH AUDIT")
    print("=" * 100)

    filename_index = build_filename_index()

    print(
        f"\nIndexed TIFF filenames: "
        f"{len(filename_index)}"
    )

    audit_rows = []

    for source_name, table_path in (
        INPUT_TABLES.items()
    ):
        dataframe = load_table(
            source_name,
            table_path,
        )

        print(
            f"\n{source_name}: "
            f"{len(dataframe)} rows"
        )

        for row_index, row in (
            dataframe.iterrows()
        ):
            record = {
                "dataset_source":
                    source_name,
                "source_table":
                    str(table_path),
                "source_row_index":
                    row_index,
                "status":
                    "pending",
                "error":
                    "",
            }

            record["overpass_id"] = (
                first_available(
                    row,
                    [
                        "overpass_id",
                        "event_id",
                    ],
                )
            )

            record["raster_group_id"] = (
                first_available(
                    row,
                    [
                        "raster_group_id",
                        "scene_group_id",
                    ],
                )
            )

            record["label"] = (
                first_available(
                    row,
                    [
                        "final_label",
                        "final_scene_label",
                        "label",
                    ],
                )
            )

            record["site_name"] = (
                first_available(
                    row,
                    [
                        "site_name",
                        "site",
                        "site_key",
                    ],
                )
            )

            record["landsat_sensor"] = (
                first_available(
                    row,
                    [
                        "landsat_sensor",
                        "sensor",
                        "SPACECRAFT_ID",
                        "gee_SPACECRAFT_ID",
                    ],
                )
            )

            record["landsat_product_id"] = (
                first_available(
                    row,
                    [
                        "LANDSAT_PRODUCT_ID",
                        "landsat_product_id",
                        "gee_LANDSAT_PRODUCT_ID",
                    ],
                )
            )

            record["acquisition_time"] = (
                first_available(
                    row,
                    [
                        "landsat_image_time_utc",
                        "landsat_image_time",
                        "candidate_time_utc",
                        "acquisition_time",
                    ],
                )
            )

            try:
                patch_path = resolve_patch_path(
                    row,
                    filename_index,
                )

                with rasterio.open(
                    patch_path
                ) as dataset:
                    array = dataset.read()

                    record.update({
                        "resolved_patch_path":
                            str(patch_path),
                        "filename":
                            patch_path.name,
                        "band_count":
                            int(dataset.count),
                        "height":
                            int(dataset.height),
                        "width":
                            int(dataset.width),
                        "dtype":
                            str(dataset.dtypes[0]),
                        "crs":
                            str(dataset.crs),
                        "nodata":
                            dataset.nodata,
                        "all_zero":
                            bool(
                                np.all(array == 0)
                            ),
                        "zero_pixel_fraction":
                            float(
                                np.mean(array == 0)
                            ),
                        "raw_dn_min":
                            float(
                                np.min(array)
                            ),
                        "raw_dn_max":
                            float(
                                np.max(array)
                            ),
                        "canonical_pixel_hash":
                            calculate_canonical_pixel_hash(
                                array
                            ),
                        "status":
                            "success",
                    })

                print(
                    f"[OK] {source_name} | "
                    f"{patch_path.name} | "
                    f"label={record['label']} | "
                    f"shape={record['height']}×"
                    f"{record['width']}"
                )

            except Exception as error:
                record["status"] = "error"
                record["error"] = str(error)

                print(
                    f"[ERROR] {source_name} | "
                    f"row={row_index} | {error}"
                )

            audit_rows.append(record)

    audit = pd.DataFrame(
        audit_rows
    )

    successful = audit[
        audit["status"] == "success"
    ].copy()

    successful[
        "duplicate_pixel_hash"
    ] = successful[
        "canonical_pixel_hash"
    ].duplicated(
        keep=False
    )

    successful[
        "duplicate_file_path"
    ] = successful[
        "resolved_patch_path"
    ].duplicated(
        keep=False
    )

    duplicate_rows = successful[
        successful[
            "duplicate_pixel_hash"
        ]
    ].copy()

    if len(duplicate_rows) > 0:
        group_map = {
            pixel_hash: (
                f"DUP_{group_number:03d}"
            )
            for group_number, pixel_hash
            in enumerate(
                sorted(
                    duplicate_rows[
                        "canonical_pixel_hash"
                    ].unique()
                ),
                start=1,
            )
        }

        duplicate_rows[
            "duplicate_group_id"
        ] = duplicate_rows[
            "canonical_pixel_hash"
        ].map(group_map)

        duplicate_rows[
            "mixed_label_group"
        ] = duplicate_rows.groupby(
            "duplicate_group_id"
        )["label"].transform(
            lambda values: (
                values.dropna()
                .astype(str)
                .nunique()
                > 1
            )
        )

        duplicate_rows[
            "mixed_source_group"
        ] = duplicate_rows.groupby(
            "duplicate_group_id"
        )["dataset_source"].transform(
            "nunique"
        ) > 1

    else:
        duplicate_rows[
            "duplicate_group_id"
        ] = pd.Series(
            dtype="object"
        )

        duplicate_rows[
            "mixed_label_group"
        ] = pd.Series(
            dtype="bool"
        )

        duplicate_rows[
            "mixed_source_group"
        ] = pd.Series(
            dtype="bool"
        )

    audit = audit.merge(
        successful[
            [
                "dataset_source",
                "source_row_index",
                "duplicate_pixel_hash",
                "duplicate_file_path",
            ]
        ],
        on=[
            "dataset_source",
            "source_row_index",
        ],
        how="left",
    )

    AUDIT_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_csv(
        AUDIT_OUTPUT,
        index=False,
    )

    duplicate_rows.to_csv(
        DUPLICATE_OUTPUT,
        index=False,
    )

    print("\n" + "=" * 100)
    print("CROSS-DATASET DUPLICATE SUMMARY")
    print("=" * 100)

    print(
        f"\nTotal input rows: "
        f"{len(audit)}"
    )

    print(
        f"Successfully hashed: "
        f"{len(successful)}"
    )

    print(
        f"Failed rows: "
        f"{int((audit['status'] != 'success').sum())}"
    )

    print("\nRows by dataset source:")
    print(
        successful[
            "dataset_source"
        ].value_counts()
    )

    print(
        f"\nUnique pixel rasters: "
        f"{successful['canonical_pixel_hash'].nunique()}"
    )

    duplicate_group_count = (
        duplicate_rows[
            "canonical_pixel_hash"
        ].nunique()
    )

    print(
        f"Duplicate pixel groups: "
        f"{duplicate_group_count}"
    )

    print(
        f"Rows involved in duplicate groups: "
        f"{len(duplicate_rows)}"
    )

    print(
        "Duplicate file paths: "
        f"{int(successful['duplicate_file_path'].sum())}"
    )

    if len(duplicate_rows) == 0:
        print(
            "\nNo cross-dataset pixel duplicates found."
        )

    else:
        print("\nDuplicate groups:")

        display_columns = [
            "duplicate_group_id",
            "dataset_source",
            "overpass_id",
            "raster_group_id",
            "label",
            "site_name",
            "landsat_product_id",
            "filename",
            "mixed_label_group",
            "mixed_source_group",
        ]

        print(
            duplicate_rows[
                display_columns
            ].sort_values(
                [
                    "duplicate_group_id",
                    "dataset_source",
                ]
            ).to_string(index=False)
        )

    print("\nSaved:")
    print(AUDIT_OUTPUT)
    print(DUPLICATE_OUTPUT)


if __name__ == "__main__":
    main()
