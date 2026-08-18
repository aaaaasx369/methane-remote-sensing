from __future__ import annotations

from pathlib import Path
import math

import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform


PROJECT = Path("/Users/happydoraaa/methane_release_project")
METHANEFUSE = Path("/Users/happydoraaa/MethaneFuse")

MANIFEST_IN = METHANEFUSE / "data/custom/exact3_s2_controlled_release.csv"
GT_IN = PROJECT / "outputs/80_s2_verified_emission_predictions.csv"
INTERVALS_IN = PROJECT / "outputs/309_all_exact_release_intervals_for_s2.csv"

CROP_DIR = PROJECT / "methanefuse_input/s2_12band_480m_exact3"
MANIFEST_OUT = METHANEFUSE / "data/custom/exact3_s2_controlled_release_480m_smoke.csv"

FOOTPRINT_M = 480.0


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"{label} 缺少欄位：{missing}\n"
            f"目前欄位：{df.columns.tolist()}"
        )


def main() -> None:
    for path in [MANIFEST_IN, GT_IN, INTERVALS_IN]:
        if not path.exists():
            raise FileNotFoundError(f"找不到：{path}")

    manifest = pd.read_csv(MANIFEST_IN, low_memory=False)
    gt = pd.read_csv(GT_IN, low_memory=False)
    intervals = pd.read_csv(INTERVALS_IN, low_memory=False)

    require_columns(
        manifest,
        [
            "id",
            "label",
            "sample_id",
            "s2_0_path",
            "s2_90_path",
            "s2_360_path",
        ],
        "MethaneFuse manifest",
    )
    require_columns(
        gt,
        ["sample_id", "release_interval_id"],
        "verified ground truth",
    )
    require_columns(
        intervals,
        ["release_interval_id", "lat", "lon"],
        "release intervals",
    )

    interval_coords = (
        intervals[
            ["release_interval_id", "lat", "lon"]
        ]
        .dropna()
        .drop_duplicates(
            subset=["release_interval_id"],
            keep="first",
        )
    )

    table = (
        manifest
        .merge(
            gt[["sample_id", "release_interval_id"]],
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            interval_coords,
            on="release_interval_id",
            how="left",
            validate="many_to_one",
        )
    )

    if table[["release_interval_id", "lat", "lon"]].isna().any().any():
        bad = table[
            table[["release_interval_id", "lat", "lon"]]
            .isna()
            .any(axis=1)
        ]
        raise ValueError(
            "以下 samples 沒有 source coordinate：\n"
            + bad[
                ["sample_id", "release_interval_id", "lat", "lon"]
            ].to_string(index=False)
        )

    CROP_DIR.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict] = []

    for _, row in table.iterrows():
        source_path = Path(str(row["s2_0_path"])).expanduser()
        if not source_path.exists():
            raise FileNotFoundError(f"找不到影像：{source_path}")

        output_path = CROP_DIR / f"{row['sample_id']}_S2_12band_480m_smoke.tif"

        with rasterio.open(source_path) as src:
            if src.crs is None:
                raise ValueError(f"影像沒有 CRS：{source_path}")
            if src.count != 12:
                raise ValueError(
                    f"預期 12 bands，但 {source_path} 有 {src.count} bands"
                )

            x_values, y_values = transform(
                "EPSG:4326",
                src.crs,
                [float(row["lon"])],
                [float(row["lat"])],
            )
            source_x = x_values[0]
            source_y = y_values[0]

            source_row, source_col = src.index(source_x, source_y)

            xres = abs(float(src.transform.a))
            yres = abs(float(src.transform.e))

            width_px = max(1, int(round(FOOTPRINT_M / xres)))
            height_px = max(1, int(round(FOOTPRINT_M / yres)))

            # Keep an exact 480 m footprint whenever the resolution divides 480 m.
            footprint_x = width_px * xres
            footprint_y = height_px * yres

            if not math.isclose(footprint_x, FOOTPRINT_M, abs_tol=1e-6):
                raise ValueError(
                    f"X resolution {xres} m cannot form an exact 480 m crop."
                )
            if not math.isclose(footprint_y, FOOTPRINT_M, abs_tol=1e-6):
                raise ValueError(
                    f"Y resolution {yres} m cannot form an exact 480 m crop."
                )

            col_off = source_col - (width_px // 2)
            row_off = source_row - (height_px // 2)

            window = Window(
                col_off=col_off,
                row_off=row_off,
                width=width_px,
                height=height_px,
            )

            inside = (
                col_off >= 0
                and row_off >= 0
                and col_off + width_px <= src.width
                and row_off + height_px <= src.height
            )

            if not inside:
                raise ValueError(
                    f"480 m crop 超出原影像範圍：{row['sample_id']}\n"
                    f"source pixel=(row={source_row}, col={source_col}), "
                    f"image=(height={src.height}, width={src.width}), "
                    f"window={window}"
                )

            data = src.read(window=window)
            profile = src.profile.copy()
            profile.update(
                height=height_px,
                width=width_px,
                transform=src.window_transform(window),
                count=12,
            )

            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(data)

        output = row.to_dict()
        resolved = str(output_path.resolve())
        output["s2_0_path"] = resolved
        output["s2_90_path"] = resolved
        output["s2_360_path"] = resolved
        output["smoke_test_only"] = True
        output["temporal_frames_identical"] = True
        output["spatial_footprint_m"] = FOOTPRINT_M
        output["source_center_lat"] = float(row["lat"])
        output["source_center_lon"] = float(row["lon"])
        output_rows.append(output)

        print(
            f"{row['sample_id']}: "
            f"source pixel=({source_row}, {source_col}), "
            f"crop={height_px}x{width_px}, "
            f"saved={output_path}"
        )

    output_df = pd.DataFrame(output_rows)

    preferred = [
        "id",
        "label",
        "sample_id",
        "metered_release_rate_kg_hr",
        "s2_0_path",
        "s2_90_path",
        "s2_360_path",
        "smoke_test_only",
        "temporal_frames_identical",
        "spatial_footprint_m",
        "source_center_lat",
        "source_center_lon",
        "release_interval_id",
    ]
    output_df = output_df[
        [column for column in preferred if column in output_df.columns]
        + [column for column in output_df.columns if column not in preferred]
    ]

    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(MANIFEST_OUT, index=False)

    print("\nCreated:")
    print(MANIFEST_OUT)
    print("\nRows:", len(output_df))


if __name__ == "__main__":
    main()
