from __future__ import annotations

from pathlib import Path
import json
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT = Path("/Users/happydoraaa/methane_release_project")
OUTPUT_DIR = PROJECT / "outputs"

INPUT_CANDIDATES = [
    OUTPUT_DIR / "40_multisite_s2_features_strict_zero_shot.csv",
    OUTPUT_DIR / "39_multisite_s2_features.csv",
]

LABEL_CANDIDATES = [
    "label",
    "physical_release_gt",
    "target",
    "class",
    "y",
]

SITE_CANDIDATES = [
    "site_id",
    "site",
    "canonical_site_id",
    "site_name",
    "location_id",
]

# These are metadata / leakage fields, not image-derived predictors.
EXACT_EXCLUDE = {
    "label",
    "physical_release_gt",
    "target",
    "class",
    "y",
    "ground_truth_status",
    "ground_truth_basis",
    "ground_truth_provenance",
    "metered_release_rate_kg_hr",
    "metered_lower_bound_kg_hr",
    "metered_upper_bound_kg_hr",
    "release_rate_kg_hr",
    "emission_rate_kg_hr",
    "site_id",
    "site",
    "canonical_site_id",
    "site_name",
    "location_id",
    "sample_id",
    "record_id",
    "scene_id",
    "image_id",
    "event_id",
    "observation_id",
    "patch_id",
    "sensor",
    "satellite",
    "source_file",
    "image_path",
    "patch_path",
    "tif_path",
    "nc_path",
    "filepath",
    "filename",
    "acquisition_time",
    "acquisition_time_utc",
    "timestamp",
    "timestamp_utc",
    "datetime",
    "date",
    "source_latitude",
    "source_longitude",
    "latitude",
    "longitude",
    "lat",
    "lon",
    "split",
    "fold",
    "prediction",
    "predicted_label",
    "probability",
    "predicted_probability",
}

SUBSTRING_EXCLUDE = [
    "ground_truth",
    "metered_release",
    "release_rate",
    "emission_rate",
    "physical_release",
    "tc_classification",
    "predicted_",
    "prediction",
    "probability",
    "_path",
    "filepath",
    "filename",
    "timestamp",
    "datetime",
    "acquisition_date",
    "acquisition_time",
    "latitude",
    "longitude",
]

RANDOM_STATE = 42


def select_existing_input() -> Path:
    for path in INPUT_CANDIDATES:
        if path.exists():
            return path
    tried = "\n".join(str(path) for path in INPUT_CANDIDATES)
    raise FileNotFoundError(
        "找不到 Sentinel-2 feature table。檢查以下路徑：\n" + tried
    )


def detect_column(columns: list[str], candidates: list[str], role: str) -> str:
    lower_lookup = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower_lookup:
            return lower_lookup[candidate.lower()]

    raise ValueError(
        f"無法自動找到 {role} 欄位。\n"
        f"候選名稱：{candidates}\n"
        f"目前欄位：{columns}"
    )


def normalize_binary_label(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().all():
        unique = set(numeric.astype(int).unique())
        if unique.issubset({0, 1}):
            return numeric.astype(int)

    mapping = {
        "0": 0,
        "1": 1,
        "negative": 0,
        "positive": 1,
        "no": 0,
        "yes": 1,
        "false": 0,
        "true": 1,
        "confirmed_no_release": 0,
        "confirmed_release": 1,
        "no_known_plume_reference": 0,
        "plume_positive": 1,
    }

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    if normalized.isna().any():
        bad = series[normalized.isna()].drop_duplicates().tolist()
        raise ValueError(f"無法轉換成 0/1 的標籤值：{bad}")

    return normalized.astype(int)


def should_exclude(column: str, label_column: str, site_column: str) -> bool:
    lower = column.lower()

    if column in {label_column, site_column}:
        return True

    if lower in EXACT_EXCLUDE:
        return True

    if any(token in lower for token in SUBSTRING_EXCLUDE):
        return True

    # Numeric identifiers are usually leakage or meaningless for generalization.
    if lower.endswith("_id") or lower.startswith("id_"):
        return True

    return False


def safe_roc_auc(y_true: pd.Series, probability: np.ndarray) -> float:
    if pd.Series(y_true).nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, probability))


def metric_row(
    y_true: pd.Series,
    y_pred: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else float("nan")

    return {
        "support": int(len(y_true)),
        "positive_support": int((y_true == 1).sum()),
        "negative_support": int((y_true == 0).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "recall_positive": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "specificity": float(specificity),
        "precision_positive": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "f1_positive": float(
            f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "roc_auc": safe_roc_auc(y_true, probability),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_path = select_existing_input()

    print(f"Input: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)

    label_column = detect_column(
        df.columns.tolist(),
        LABEL_CANDIDATES,
        "label",
    )
    site_column = detect_column(
        df.columns.tolist(),
        SITE_CANDIDATES,
        "site",
    )

    df = df.copy()
    df["_target"] = normalize_binary_label(df[label_column])
    df["_site"] = df[site_column].astype(str).str.strip()

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    feature_columns = [
        column
        for column in numeric_columns
        if column != "_target"
        and not should_exclude(column, label_column, site_column)
    ]

    # Remove all-missing and constant columns before cross-validation.
    all_missing = [
        column for column in feature_columns if df[column].isna().all()
    ]
    feature_columns = [
        column for column in feature_columns if column not in all_missing
    ]

    constant_columns = [
        column
        for column in feature_columns
        if df[column].nunique(dropna=True) <= 1
    ]
    feature_columns = [
        column for column in feature_columns if column not in constant_columns
    ]

    if not feature_columns:
        raise ValueError("沒有可用的 numeric feature columns。")

    X = df[feature_columns].replace([np.inf, -np.inf], np.nan)
    y = df["_target"]
    groups = df["_site"]

    audit = (
        df.groupby(["_site", "_target"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "negative", 1: "positive"})
        .reset_index()
        .rename(columns={"_site": "site_id"})
    )
    audit["total"] = audit.get("negative", 0) + audit.get("positive", 0)
    audit_path = OUTPUT_DIR / "75_s2_five_site_audit.csv"
    audit.to_csv(audit_path, index=False)

    selected_path = OUTPUT_DIR / "75_s2_selected_feature_columns.txt"
    selected_path.write_text(
        "\n".join(
            [
                f"input={input_path}",
                f"label_column={label_column}",
                f"site_column={site_column}",
                f"rows={len(df)}",
                f"sites={groups.nunique()}",
                f"features={len(feature_columns)}",
                "",
                "[selected_features]",
                *feature_columns,
                "",
                "[dropped_all_missing]",
                *all_missing,
                "",
                "[dropped_constant]",
                *constant_columns,
            ]
        ),
        encoding="utf-8",
    )

    print("\nSite/label audit:")
    print(audit.to_string(index=False))
    print(f"\nSelected numeric features: {len(feature_columns)}")

    if len(df) != 75:
        warnings.warn(f"預期 75 rows，但讀到 {len(df)} rows。")

    if groups.nunique() != 5:
        warnings.warn(f"預期 5 sites，但讀到 {groups.nunique()} sites。")

    preprocessing = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    models = {
        "dummy_prior": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", DummyClassifier(strategy="prior")),
            ]
        ),
        "logistic_balanced": Pipeline(
            steps=[
                ("preprocess", preprocessing),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest_balanced": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=5,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    logo = LeaveOneGroupOut()
    fold_rows: list[dict] = []
    prediction_rows: list[dict] = []

    for fold_index, (train_index, test_index) in enumerate(
        logo.split(X, y, groups),
        start=1,
    ):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        train_sites = sorted(groups.iloc[train_index].unique().tolist())
        test_sites = sorted(groups.iloc[test_index].unique().tolist())

        if len(test_sites) != 1:
            raise RuntimeError(
                f"LeaveOneGroupOut test fold 應只有 1 site，實際為 {test_sites}"
            )

        held_out_site = test_sites[0]

        for model_name, estimator in models.items():
            model = clone(estimator)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test).astype(int)

            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(X_test)[:, 1]
            else:
                score = model.decision_function(X_test)
                probability = 1.0 / (1.0 + np.exp(-score))

            row = {
                "fold": fold_index,
                "model": model_name,
                "held_out_site": held_out_site,
                "train_sites": " | ".join(train_sites),
                "train_rows": int(len(train_index)),
                "test_rows": int(len(test_index)),
                **metric_row(y_test, y_pred, probability),
            }
            fold_rows.append(row)

            metadata_candidates = [
                "sample_id",
                "record_id",
                "patch_id",
                "scene_id",
                "image_id",
                "image_path",
                "patch_path",
                "tif_path",
                "acquisition_time_utc",
                "timestamp_utc",
            ]
            metadata_columns = [
                column
                for column in metadata_candidates
                if column in df.columns
            ]

            for position, row_index in enumerate(test_index):
                pred_row = {
                    "row_index": int(row_index),
                    "fold": fold_index,
                    "model": model_name,
                    "held_out_site": held_out_site,
                    "true_label": int(y.iloc[row_index]),
                    "predicted_label": int(y_pred[position]),
                    "positive_probability": float(probability[position]),
                }

                for column in metadata_columns:
                    pred_row[column] = df.iloc[row_index][column]

                prediction_rows.append(pred_row)

    fold_metrics = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)

    fold_path = OUTPUT_DIR / "76_s2_loso_fold_metrics.csv"
    pred_path = OUTPUT_DIR / "77_s2_loso_oof_predictions.csv"
    summary_path = OUTPUT_DIR / "78_s2_loso_summary.csv"

    fold_metrics.to_csv(fold_path, index=False)
    predictions.to_csv(pred_path, index=False)

    summary_rows = []

    for model_name, part in predictions.groupby("model"):
        metrics = metric_row(
            part["true_label"],
            part["predicted_label"].to_numpy(),
            part["positive_probability"].to_numpy(),
        )
        metrics["model"] = model_name
        metrics["sites_tested"] = int(part["held_out_site"].nunique())
        metrics["rows"] = int(len(part))
        summary_rows.append(metrics)

    summary = pd.DataFrame(summary_rows)
    summary = summary[
        [
            "model",
            "sites_tested",
            "rows",
            "support",
            "positive_support",
            "negative_support",
            "tn",
            "fp",
            "fn",
            "tp",
            "accuracy",
            "balanced_accuracy",
            "recall_positive",
            "specificity",
            "precision_positive",
            "f1_positive",
            "roc_auc",
        ]
    ].sort_values(
        ["balanced_accuracy", "roc_auc"],
        ascending=False,
        na_position="last",
    )

    summary.to_csv(summary_path, index=False)

    print("\nOverall out-of-site summary:")
    print(summary.to_string(index=False))

    print("\nCreated:")
    for path in [
        audit_path,
        selected_path,
        fold_path,
        pred_path,
        summary_path,
    ]:
        print(path)


if __name__ == "__main__":
    main()
