from pathlib import Path
import rasterio
import numpy as np
import matplotlib.pyplot as plt

patch_dir = Path("sample_patches/methaneair_s2")
out_dir = Path("sample_patches/methaneair_s2_preview")
out_dir.mkdir(parents=True, exist_ok=True)

def normalize_band(band):
    band = band.astype(float)
    p2, p98 = np.percentile(band[band > 0], [2, 98])
    band = np.clip((band - p2) / (p98 - p2), 0, 1)
    return band

for tif_path in sorted(patch_dir.glob("*.tif")):
    with rasterio.open(tif_path) as src:
        # 你匯出的 bands 順序是：
        # B2, B3, B4, B8, B11, B12
        blue = src.read(1)
        green = src.read(2)
        red = src.read(3)

        rgb = np.dstack([
            normalize_band(red),
            normalize_band(green),
            normalize_band(blue)
        ])

        plt.figure(figsize=(5, 5))
        plt.imshow(rgb)
        plt.axis("off")
        plt.title(tif_path.stem)

        out_path = out_dir / f"{tif_path.stem}_rgb.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()

        print("Saved:", out_path)