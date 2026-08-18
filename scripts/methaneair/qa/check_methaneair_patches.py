from pathlib import Path
import pandas as pd
import rasterio
import numpy as np

patch_dir = Path("sample_patches/methaneair_s2")
out_csv = Path("outputs/17_methaneair_s2_patch_quality.csv")

rows = []

for tif_path in sorted(patch_dir.glob("*.tif")):
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
                "has_nan": bool(np.isnan(arr.astype(float)).any())
            })

    except Exception as e:
        rows.append({
            "filename": tif_path.name,
            "error": str(e)
        })

quality = pd.DataFrame(rows)
quality.to_csv(out_csv, index=False)

print(quality)
print("Saved:", out_csv)