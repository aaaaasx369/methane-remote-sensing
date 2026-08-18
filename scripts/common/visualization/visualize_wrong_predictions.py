from pathlib import Path
import pandas as pd
import numpy as np
import rasterio
import matplotlib.pyplot as plt


INPUT_PATH = Path("outputs/27_wrong_predictions.csv")
OUT_DIR = Path("outputs/wrong_prediction_previews")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_band(band):
    band = band.astype(float)
    valid = band[np.isfinite(band)]

    if len(valid) == 0:
        return np.zeros_like(band, dtype=float)

    p2, p98 = np.percentile(valid, [2, 98])

    if p98 == p2:
        return np.zeros_like(band, dtype=float)

    return np.clip((band - p2) / (p98 - p2), 0, 1)


def make_rgb(tif_path):
    with rasterio.open(tif_path) as src:
        # bands order: B2, B3, B4, B8, B11, B12
        blue = src.read(1)
        green = src.read(2)
        red = src.read(3)

    rgb = np.dstack([
        normalize_band(red),
        normalize_band(green),
        normalize_band(blue),
    ])

    return rgb


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_PATH}. "
            "Please generate outputs/27_wrong_predictions.csv first."
        )

    df = pd.read_csv(INPUT_PATH)

    print("Wrong predictions:", len(df))

    for _, row in df.iterrows():
        tif_path = Path(row["relative_path"])

        if not tif_path.exists():
            print("Missing:", tif_path)
            continue

        rgb = make_rgb(tif_path)

        label = int(row["label"])
        pred = int(row["pred_label"])
        prob = float(row["pred_prob_positive"])
        split = row["eval_split"]

        title = (
            f"{row['filename']}\n"
            f"split={split}, true={label}, pred={pred}, prob={prob:.3f}"
        )

        plt.figure(figsize=(5, 5))
        plt.imshow(rgb)
        plt.axis("off")
        plt.title(title)

        out_name = (
            f"{split}_true{label}_pred{pred}_prob{prob:.2f}_"
            f"{Path(row['filename']).stem}.png"
        )

        out_path = OUT_DIR / out_name
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()

        print("Saved:", out_path)

    print("Done. Preview folder:", OUT_DIR)


if __name__ == "__main__":
    main()
