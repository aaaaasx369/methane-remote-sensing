from pathlib import Path
import pandas as pd
import numpy as np


INPUT_PATH = Path("outputs/23_combined_classification_dataset.csv")
OUT_PATH = Path("outputs/24_experiment_split.csv")

RANDOM_SEED = 42


def stratified_split(df, label_col="label", train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Stratified split by label.
    """
    rng = np.random.default_rng(seed)

    df = df.copy()
    df["split"] = ""

    for label, sub in df.groupby(label_col):
        idx = sub.index.to_numpy()
        rng.shuffle(idx)

        n = len(idx)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))

        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]

        df.loc[train_idx, "split"] = "train"
        df.loc[val_idx, "split"] = "val"
        df.loc[test_idx, "split"] = "test"

    return df


def main():
    df = pd.read_csv(INPUT_PATH)

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)

    cr = df[df["dataset_group"] == "controlled_release_s2"].copy()
    ma = df[df["dataset_group"] == "methaneair_s2_positive"].copy()

    # Controlled release 有 label 0 / 1，適合切 train / val / test
    cr_split = stratified_split(
        cr,
        label_col="label",
        train_ratio=0.7,
        val_ratio=0.15,
        seed=RANDOM_SEED
    )

    # MethaneAIR 只有 positive，先全部放進 train，當 additional positive training data
    ma["split"] = "train"

    combined_split = pd.concat([cr_split, ma], ignore_index=True, sort=False)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined_split.to_csv(OUT_PATH, index=False)

    print("Saved:", OUT_PATH)
    print("Total samples:", len(combined_split))

    print("\nSplit counts:")
    print(combined_split["split"].value_counts())

    print("\nLabel counts by split:")
    print(pd.crosstab(combined_split["split"], combined_split["label"]))

    print("\nDataset group by split:")
    print(pd.crosstab(combined_split["split"], combined_split["dataset_group"]))

    print("\nDataset group + label by split:")
    print(pd.crosstab(
        [combined_split["split"], combined_split["dataset_group"]],
        combined_split["label"]
    ))


if __name__ == "__main__":
    main()