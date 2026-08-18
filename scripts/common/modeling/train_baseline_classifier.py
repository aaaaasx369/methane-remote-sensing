from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


INPUT_PATH = Path("outputs/25_s2_patch_features.csv")
OUT_PRED_PATH = Path("outputs/26_baseline_predictions.csv")


META_COLS = [
    "filename",
    "relative_path",
    "label",
    "split",
    "dataset_group",
    "event_id",
    "ground_truth_type",
    "sensor",
    "datetime_utc",
    "lat",
    "lon",
]


def evaluate(y_true, y_pred, name):
    print(f"\n========== {name} ==========")

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Balanced accuracy:", balanced_accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, zero_division=0))
    print("F1:", f1_score(y_true, y_pred, zero_division=0))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, zero_division=0))


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_PATH}. Please run extract_s2_patch_features.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)

    feature_cols = [
        c for c in df.columns
        if c not in META_COLS
    ]

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    print("Total samples:", len(df))
    print("Train:", len(train_df))
    print("Val:", len(val_df))
    print("Test:", len(test_df))

    print("\nLabel counts by split:")
    print(pd.crosstab(df["split"], df["label"]))

    print("\nDataset group by split:")
    print(pd.crosstab(df["split"], df["dataset_group"]))

    X_train = train_df[feature_cols]
    y_train = train_df["label"]

    X_val = val_df[feature_cols]
    y_val = val_df["label"]

    X_test = test_df[feature_cols]
    y_test = test_df["label"]

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            max_depth=5
        ))
    ])

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    evaluate(y_val, val_pred, "Validation")
    evaluate(y_test, test_pred, "Test")

    pred_rows = []

    for split_name, sub_df, X_sub in [
        ("val", val_df, X_val),
        ("test", test_df, X_test)
    ]:
        pred = model.predict(X_sub)
        prob = model.predict_proba(X_sub)[:, 1]

        keep_cols = [c for c in META_COLS if c in sub_df.columns]
        out = sub_df[keep_cols].copy()
        out["pred_label"] = pred
        out["pred_prob_positive"] = prob
        out["eval_split"] = split_name
        pred_rows.append(out)

    pred_df = pd.concat(pred_rows, ignore_index=True)
    OUT_PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(OUT_PRED_PATH, index=False)

    print("\nSaved predictions:", OUT_PRED_PATH)


if __name__ == "__main__":
    main()
