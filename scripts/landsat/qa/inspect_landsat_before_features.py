from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


INPUT_CSV = Path(
    "outputs/32_controlled_release_landsat_dataset_table.csv"
)

OUTPUT_CSV = Path(
    "outputs/landsat_pre_feature_audit.csv"
)

PATH_COLUMN_CANDIDATES = [
    "patch_path",
    "tif_path",
    "file_path",
    "filepath",
    "local_path",
    "image_path",
    "output_path",
    "patch_file",
    "filename",
    "file_name",
    "output_filename",
]


def build_tif_index():
    """
    搜尋專案內可能存放 Landsat TIFF 的資料夾。
    """
    search_directories = [
        Path("outputs"),
        Path("sample_patches"),
        Path("patches"),
        Path("data"),
        Path("downloads"),
    ]

    tif_index = {}

    for directory in search_directories:
        if not directory.exists():
            continue

        for extension in ("*.tif", "*.tiff"):
            for path in directory.rglob(extension):
                tif_index.setdefault(path.name, []).append(path)

    return tif_index


def resolve_patch_path(row, tif_index):
    """
    從 CSV 的一列資料中尋找影像路徑。
    """
    possible_values = []

    # 先檢查常見的路徑欄位
    for column in PATH_COLUMN_CANDIDATES:
        if column in row.index and pd.notna(row[column]):
            possible_values.append(str(row[column]).strip())

    # 再檢查所有欄位中是否有 .tif 或 .tiff 字串
    for value in row.values:
        if isinstance(value, str):
            value = value.strip()

            if value.lower().endswith((".tif", ".tiff")):
                possible_values.append(value)

    # 去除重複值
    possible_values = list(dict.fromkeys(possible_values))

    for value in possible_values:
        path = Path(value).expanduser()

        # 完整路徑或目前工作目錄下的相對路徑
        if path.exists():
            return path.resolve()

        project_relative = Path.cwd() / path

        if project_relative.exists():
            return project_relative.resolve()

        # CSV 可能只存檔名
        matches = tif_index.get(path.name, [])

        if len(matches) == 1:
            return matches[0].resolve()

        if len(matches) > 1:
            print(
                f"[WARNING] 找到多個同名檔案：{path.name}"
            )
            print(f"          暫時使用：{matches[0]}")

            return matches[0].resolve()

    return None


def get_band_statistics(src, band_number):
    """
    計算單一波段的有效像素統計。
    """
    band = src.read(
        band_number,
        masked=True,
    ).astype(np.float64)

    values = band.compressed()
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            "valid_pixels": 0,
            "minimum": np.nan,
            "maximum": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "zero_fraction": np.nan,
        }

    return {
        "valid_pixels": int(len(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "zero_fraction": float(np.mean(values == 0)),
    }


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"找不到輸入 CSV：{INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)

    print("=" * 80)
    print("LANDSAT PRE-FEATURE AUDIT")
    print("=" * 80)

    print(f"\nCSV 路徑：{INPUT_CSV}")
    print(f"CSV rows：{len(df)}")
    print(f"CSV columns：{len(df.columns)}")

    print("\n所有 CSV 欄位：")
    for index, column in enumerate(df.columns, start=1):
        print(f"{index:02d}. {column}")

    print("\n前 3 列資料：")
    print(df.head(3).to_string())

    tif_index = build_tif_index()

    print(
        f"\n專案內找到的唯一 TIFF 檔名數量："
        f"{len(tif_index)}"
    )

    audit_rows = []
    unresolved_rows = []

    for row_index, row in df.iterrows():
        patch_path = resolve_patch_path(
            row,
            tif_index,
        )

        if patch_path is None:
            unresolved_rows.append(row_index)

            audit_rows.append({
                "row_index": row_index,
                "resolved": False,
                "patch_path": None,
            })

            print(
                f"[NOT FOUND] row={row_index}"
            )

            continue

        try:
            with rasterio.open(patch_path) as src:
                descriptions = list(src.descriptions)

                audit_row = {
                    "row_index": row_index,
                    "resolved": True,
                    "patch_path": str(patch_path),
                    "filename": patch_path.name,
                    "band_count": src.count,
                    "height": src.height,
                    "width": src.width,
                    "dtype": str(src.dtypes[0]),
                    "nodata": src.nodata,
                    "crs": str(src.crs),
                    "pixel_width": abs(src.transform.a),
                    "pixel_height": abs(src.transform.e),
                    "band_descriptions": str(descriptions),
                }

                # 每個波段都計算基本統計
                for band_number in range(
                    1,
                    src.count + 1,
                ):
                    stats = get_band_statistics(
                        src,
                        band_number,
                    )

                    for stat_name, stat_value in stats.items():
                        audit_row[
                            f"band_{band_number}_{stat_name}"
                        ] = stat_value

                audit_rows.append(audit_row)

                print(
                    f"[OK] {row_index + 1:02d}/{len(df)} "
                    f"{patch_path.name} | "
                    f"bands={src.count} | "
                    f"shape={src.height}×{src.width} | "
                    f"dtype={src.dtypes[0]}"
                )

                # 詳細顯示前 3 張影像
                if row_index < 3:
                    print("\n  Detailed raster information")
                    print(f"  Path: {patch_path}")
                    print(f"  CRS: {src.crs}")
                    print(f"  Transform: {src.transform}")
                    print(f"  Bounds: {src.bounds}")
                    print(f"  NoData: {src.nodata}")
                    print(
                        f"  Band descriptions: "
                        f"{descriptions}"
                    )

                    print(
                        f"  Dataset tags: "
                        f"{src.tags()}"
                    )

                    for band_number in range(
                        1,
                        src.count + 1,
                    ):
                        stats = get_band_statistics(
                            src,
                            band_number,
                        )

                        print(
                            f"  Band {band_number}: "
                            f"description="
                            f"{src.descriptions[band_number - 1]}, "
                            f"tags={src.tags(band_number)}"
                        )

                        print(
                            f"          min={stats['minimum']:.6g}, "
                            f"max={stats['maximum']:.6g}, "
                            f"mean={stats['mean']:.6g}, "
                            f"std={stats['std']:.6g}, "
                            f"zero_fraction="
                            f"{stats['zero_fraction']:.4f}"
                        )

                    print()

        except Exception as error:
            print(
                f"[ERROR] row={row_index}, "
                f"file={patch_path.name}, "
                f"error={error}"
            )

            audit_rows.append({
                "row_index": row_index,
                "resolved": True,
                "patch_path": str(patch_path),
                "filename": patch_path.name,
                "read_error": str(error),
            })

    audit_df = pd.DataFrame(audit_rows)

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    successful_df = audit_df[
        audit_df.get("band_count").notna()
    ].copy()

    print("\n" + "=" * 80)
    print("AUDIT SUMMARY")
    print("=" * 80)

    print(f"\nCSV rows：{len(df)}")
    print(
        f"成功找到並讀取的影像："
        f"{len(successful_df)}"
    )
    print(
        f"找不到影像的 rows："
        f"{len(unresolved_rows)}"
    )

    if len(successful_df) > 0:
        print("\nBand count 分布：")
        print(
            successful_df["band_count"]
            .value_counts(dropna=False)
            .sort_index()
        )

        print("\n影像尺寸分布：")
        shape_counts = (
            successful_df
            .groupby(["height", "width"])
            .size()
            .sort_values(ascending=False)
        )
        print(shape_counts)

        print("\n資料型態 dtype 分布：")
        print(
            successful_df["dtype"]
            .value_counts(dropna=False)
        )

        print("\nNoData 分布：")
        print(
            successful_df["nodata"]
            .astype(str)
            .value_counts(dropna=False)
        )

        print("\nCRS 分布：")
        print(
            successful_df["crs"]
            .value_counts(dropna=False)
        )

        print("\n像素解析度分布：")
        resolution_counts = (
            successful_df
            .groupby(["pixel_width", "pixel_height"])
            .size()
            .sort_values(ascending=False)
        )
        print(resolution_counts)

        if "landsat_sensor" in df.columns:
            print("\n原始 CSV sensor 分布：")
            print(
                df["landsat_sensor"]
                .value_counts(dropna=False)
            )

        if "label" in df.columns:
            print("\n原始 CSV label 分布：")
            print(
                df["label"]
                .value_counts(dropna=False)
                .sort_index()
            )

        print("\n每個波段的整體數值範圍：")

        band_counts = successful_df[
            "band_count"
        ].dropna()

        if len(band_counts) > 0:
            maximum_band_count = int(
                band_counts.max()
            )

            for band_number in range(
                1,
                maximum_band_count + 1,
            ):
                minimum_column = (
                    f"band_{band_number}_minimum"
                )
                maximum_column = (
                    f"band_{band_number}_maximum"
                )
                mean_column = (
                    f"band_{band_number}_mean"
                )

                if minimum_column not in successful_df.columns:
                    continue

                print(
                    f"Band {band_number}: "
                    f"global_min="
                    f"{successful_df[minimum_column].min():.6g}, "
                    f"global_max="
                    f"{successful_df[maximum_column].max():.6g}, "
                    f"average_patch_mean="
                    f"{successful_df[mean_column].mean():.6g}"
                )

    print(f"\nAudit CSV 已儲存：{OUTPUT_CSV}")


if __name__ == "__main__":
    main()
