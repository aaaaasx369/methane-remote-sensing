from pathlib import Path
import pandas as pd
import numpy as np
import rasterio
import matplotlib.pyplot as plt


INPUT_PATH = Path("outputs/27_wrong_predictions.csv")
OUT_DIR = Path("outputs/wrong_prediction_swir_previews")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize(x):
    x = x.astype(float)
    valid = x[np.isfinite(x)]

    if len(valid) == 0:
        return np.zeros_like(x)

    p2, p98 = np.percentile(valid, [2, 98])

    if p98 == p2:
        return np.zeros_like(x)

    return np.clip((x - p2) / (p98 - p2), 0, 1)


def main():
    df = pd.read_csv(INPUT_PATH)

    for _, row in df.iterrows():
        tif_path = Path(row["relative_path"])

        if not tif_path.exists():
            print("Missing:", tif_path)
            continue

        with rasterio.open(tif_path) as src:
            # Band order: B2, B3, B4, B8, B11, B12
            B2 = src.read(1).astype(float)
            B3 = src.read(2).astype(float)
            B4 = src.read(3).astype(float)
            B8 = src.read(4).astype(float)
            B11 = src.read(5).astype(float)
            B12 = src.read(6).astype(float)

        eps = 1e-6

        rgb = np.dstack([
            normalize(B4),
            normalize(B3),
            normalize(B2)
        ])

        swir_false_color = np.dstack([
            normalize(B12),
            normalize(B11),
            normalize(B8)
        ])

        ratio_b12_b11 = B12 / (B11 + eps)
        diff_b12_b11 = B12 - B11

        label = int(row["label"])
        pred = int(row["pred_label"])
        prob = float(row["pred_prob_positive"])
        split = row["eval_split"]

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title("RGB: B4/B3/B2")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(swir_false_color)
        axes[0, 1].set_title("SWIR false color: B12/B11/B8")
        axes[0, 1].axis("off")

        im1 = axes[1, 0].imshow(normalize(ratio_b12_b11))
        axes[1, 0].set_title("Ratio: B12 / B11")
        axes[1, 0].axis("off")

        im2 = axes[1, 1].imshow(normalize(diff_b12_b11))
        axes[1, 1].set_title("Difference: B12 - B11")
        axes[1, 1].axis("off")

        fig.suptitle(
            f"{row['filename']}\n"
            f"split={split}, true={label}, pred={pred}, prob={prob:.3f}",
            fontsize=14
        )

        out_name = (
            f"{split}_true{label}_pred{pred}_prob{prob:.2f}_"
            f"{Path(row['filename']).stem}_swir.png"
        )

        out_path = OUT_DIR / out_name
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()

        print("Saved:", out_path)

    print("Done:", OUT_DIR)


if __name__ == "__main__":
    main()