from pathlib import Path
import pandas as pd
import rasterio
import numpy as np


PATCH_DIR = Path("sample_patches/controlled_release_s2")
OUT_CSV = Path("outputs/21_controlled_release_s2_patch_quality.csv")

rows = []

for tif_path in sorted(PATCH_DIR.glob("*.tif")):
    try:
        with rasterio.open(tif_path) as src:
            arr = src.read()

            rows.append({
                "filename": tif_path.name,
                "width": src.width,
                "height": src.height,
                "band_count": src.count,
                "crs": str(src.crs),
                "dtype": str(arr.dtype),
                "min_value": float(np.nanmin(arr)),
                "max_value": float(np.nanmax(arr)),
                "mean_value": float(np.nanmean(arr)),
                "all_zero": bool(np.all(arr == 0)),
                "has_nan": bool(np.isnan(arr.astype(float)).any()),
            })

    except Exception as e:
        rows.append({
            "filename": tif_path.name,
            "error": str(e),
        })

quality_df = pd.DataFrame(rows)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
quality_df.to_csv(OUT_CSV, index=False)

print(quality_df)
print("Saved:", OUT_CSV)
print("Number of patches checked:", len(quality_df))