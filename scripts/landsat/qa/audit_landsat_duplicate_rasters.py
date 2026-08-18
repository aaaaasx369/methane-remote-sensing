from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import rasterio


INPUT_CSV = Path(
    "outputs/35_landsat_patch_features.csv"
)

ROW_OUTPUT_CSV = Path(
    "outputs/37_landsat_raster_duplicate_rows.csv"
)

GROUP_OUTPUT_CSV = Path(
    "outputs/38_landsat_raster_duplicate_groups.csv"
)


def hash_raster(path):
    """
    Calculate two hashes:

    pixel_hash:
        Same pixel values and masks produce the same hash.
        This is the most important hash for detecting ML leakage.

    full_raster_hash:
        Also includes CRS, transform, shape, dtype, and NoData.
    """
    with rasterio.open(path) as src:
        data = src.read()
        masks = src.read_masks()

        metadata = {
            "shape": list(data.shape),
            "dtype": [str(dtype) for dtype in src.dtypes],
            "crs": str(src.crs),
            "transform": list(src.transform),
            "nodata": src.nodata,
            "width": src.width,
            "height": src.height,
            "band_count": src.count,
        }

    data = np.ascontiguousarray(data)
    masks = np.ascontiguousarray(masks)

    pixel_hasher = hashlib.sha256()
    pixel_hasher.update(str(data.shape).encode("utf-8"))
    pixel_hasher.update(str(data.dtype).encode("utf-8"))
    pixel_hasher.update(data.tobytes())
    pixel_hasher.update(masks.tobytes())

    pixel_hash = pixel_hasher.hexdigest()

    full_hasher = hashlib.sha256()
    full_hasher.update(pixel_hash.encode("utf-8"))
    full_hasher.update(
        json.dumps(
            metadata,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    )

    full_raster_hash = full_hasher.hexdigest()

    return {
        "pixel_hash": pixel_hash,
        "full_raster_hash": full_raster_hash,
        "raster_height_hashcheck": metadata["height"],
        "raster_width_hashcheck": metadata["width"],
        "raster_band_count_hashcheck": metadata["band_count"],
        "raster_crs_hashcheck": metadata["crs"],
    }


def join_unique(series):
    values = (
        series.dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    return " | ".join(values)


def find_metadata_columns(df):
    """
    Find columns that may identify the Landsat acquisition or scene.
    """
    keywords = (
        "image_id",
        "image_time",
        "acquisition",
        "scene",
        "system_index",
        "landsat_time",
        "landsat_image",
        "date",
    )

    return [
        column
        for column in df.columns
        if any(
            keyword in column.lower()
            for keyword in keywords
        )
    ]


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)

    if "resolved_patch_path" not in df.columns:
        raise ValueError(
            "resolved_patch_path column is missing."
        )

    if "label" not in df.columns:
        raise ValueError(
            "label column is missing."
        )

    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce",
    )

    print("=" * 80)
    print("LANDSAT DUPLICATE RASTER AUDIT")
    print("=" * 80)

    print(f"\nInput rows: {len(df)}")

    hash_rows = []

    for row_index, row in df.iterrows():
        path = Path(
            str(row["resolved_patch_path"])
        ).expanduser()

        if not path.exists():
            print(
                f"[NOT FOUND] row={row_index}: {path}"
            )

            hash_rows.append({
                "row_index": row_index,
                "hash_status": "not_found",
            })

            continue

        try:
            hash_result = hash_raster(path)

            hash_result.update({
                "row_index": row_index,
                "hash_status": "success",
            })

            hash_rows.append(hash_result)

            print(
                f"[OK] {row_index + 1:02d}/{len(df)} | "
                f"{path.name} | "
                f"pixel_hash={hash_result['pixel_hash'][:12]}"
            )

        except Exception as error:
            print(
                f"[ERROR] row={row_index} | "
                f"{path.name} | {error}"
            )

            hash_rows.append({
                "row_index": row_index,
                "hash_status": "error",
                "hash_error": str(error),
            })

    hash_df = pd.DataFrame(hash_rows)

    result_df = df.reset_index(
        drop=True
    ).reset_index(
        names="row_index"
    ).merge(
        hash_df,
        on="row_index",
        how="left",
    )

    valid_df = result_df[
        result_df["hash_status"] == "success"
    ].copy()

    # Same pixel hash means the ML model sees exactly the same raster.
    valid_df["raster_group_id"] = (
        "RG_"
        + valid_df["pixel_hash"].str[:12]
    )

    group_size_map = (
        valid_df.groupby("pixel_hash")
        .size()
    )

    valid_df["raster_group_size"] = (
        valid_df["pixel_hash"]
        .map(group_size_map)
    )

    valid_df["is_duplicated_raster"] = (
        valid_df["raster_group_size"] > 1
    )

    label_count_map = (
        valid_df.groupby("pixel_hash")["label"]
        .nunique(dropna=True)
    )

    valid_df["raster_group_label_count"] = (
        valid_df["pixel_hash"]
        .map(label_count_map)
    )

    valid_df["raster_group_has_mixed_labels"] = (
        valid_df["raster_group_label_count"] > 1
    )

    metadata_columns = find_metadata_columns(
        valid_df
    )

    group_rows = []

    for pixel_hash, group in valid_df.groupby(
        "pixel_hash",
        sort=False,
    ):
        labels = sorted(
            group["label"]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        group_row = {
            "raster_group_id":
                f"RG_{pixel_hash[:12]}",
            "pixel_hash": pixel_hash,
            "n_rows": len(group),
            "n_unique_labels": len(labels),
            "labels":
                ",".join(map(str, labels)),
            "mixed_labels": len(labels) > 1,
            "n_unique_full_raster_hashes":
                group["full_raster_hash"].nunique(),
        }

        for column in [
            "filename",
            "event_id",
            "site_name",
            "landsat_sensor",
            "resolved_patch_path",
        ]:
            if column in group.columns:
                group_row[column] = join_unique(
                    group[column]
                )

        for column in metadata_columns:
            if column in group_row:
                continue

            group_row[column] = join_unique(
                group[column]
            )

        group_rows.append(group_row)

    group_df = pd.DataFrame(group_rows)

    group_df = group_df.sort_values(
        by=[
            "mixed_labels",
            "n_rows",
        ],
        ascending=[
            False,
            False,
        ],
    )

    ROW_OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    valid_df.to_csv(
        ROW_OUTPUT_CSV,
        index=False,
    )

    group_df.to_csv(
        GROUP_OUTPUT_CSV,
        index=False,
    )

    duplicated_groups = group_df[
        group_df["n_rows"] > 1
    ]

    mixed_label_groups = group_df[
        group_df["mixed_labels"]
    ]

    rows_in_duplicate_groups = int(
        valid_df[
            "is_duplicated_raster"
        ].sum()
    )

    rows_in_mixed_label_groups = int(
        valid_df[
            "raster_group_has_mixed_labels"
        ].sum()
    )

    print("\n" + "=" * 80)
    print("DUPLICATE RASTER SUMMARY")
    print("=" * 80)

    print(f"\nSuccessfully hashed rows: {len(valid_df)}")
    print(
        f"Unique pixel rasters: "
        f"{valid_df['pixel_hash'].nunique()}"
    )
    print(
        f"Unique full rasters: "
        f"{valid_df['full_raster_hash'].nunique()}"
    )
    print(
        f"Duplicate raster groups: "
        f"{len(duplicated_groups)}"
    )
    print(
        f"Rows inside duplicate groups: "
        f"{rows_in_duplicate_groups}"
    )
    print(
        f"Mixed-label raster groups: "
        f"{len(mixed_label_groups)}"
    )
    print(
        f"Rows inside mixed-label groups: "
        f"{rows_in_mixed_label_groups}"
    )

    print("\nLargest raster groups:")
    display_columns = [
        column
        for column in [
            "raster_group_id",
            "n_rows",
            "labels",
            "mixed_labels",
            "landsat_sensor",
            "site_name",
            "filename",
        ]
        if column in group_df.columns
    ]

    print(
        group_df[
            display_columns
        ].head(15).to_string(
            index=False
        )
    )

    if len(mixed_label_groups) > 0:
        print("\nWARNING: IDENTICAL RASTERS WITH DIFFERENT LABELS")
        print(
            mixed_label_groups[
                display_columns
            ].to_string(
                index=False
            )
        )

    print("\nSaved:")
    print(ROW_OUTPUT_CSV)
    print(GROUP_OUTPUT_CSV)


if __name__ == "__main__":
    main()
