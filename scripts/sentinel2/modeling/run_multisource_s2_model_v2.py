#!/usr/bin/env python3
"""
Multisource Sentinel-2 model v2
================================

Research question:
Can Sentinel-2 controlled-release data from different sources/sites be fitted
in one model, and can the model generalize to an unseen site or unseen source?

This single script:
1. Auto-finds a suitable CSV in <project_root>/outputs.
2. Auto-detects metadata columns.
3. Uses existing numeric features when available.
4. Otherwise extracts simple B11/B12/context features from GeoTIFF patches.
5. Runs:
   - grouped random split by scene (optimistic reference),
   - leave-one-site-out,
   - leave-one-source-out.
6. Writes results to outputs/500-505 and figures/506.

Typical use:
    python run_multisource_s2_model_v2.py \
      --project-root /Users/happydoraaa/methane_release_project

Self-test:
    python run_multisource_s2_model_v2.py --self-test
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        roc_auc_score,
    )
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise SystemExit(
        "Missing packages. Run:\n"
        "python -m pip install pandas numpy scikit-learn matplotlib rasterio\n"
        f"Original error: {exc}"
    ) from exc

try:
    import rasterio
except ImportError:
    rasterio = None


RANDOM_STATE = 42
EPS = 1e-9

# Prefer the most curated/newest files first.
PREFERRED_INPUTS = (
    "452_s2_multisite_features_v1.csv",
    "390_multisensor_master_manifest_v1.csv",
    "25_s2_patch_features.csv",
    "22_controlled_release_s2_dataset_table.csv",
    "20_controlled_release_s2_patch_index.csv",
)

LABEL_ALIASES = (
    "label", "final_label", "classification_label", "y", "target",
)
SITE_ALIASES = (
    "site_id", "site", "site_name", "facility", "location", "release_site",
)
SOURCE_ALIASES = (
    "source_origin", "ground_truth_source", "source_dataset", "dataset_group",
    "campaign_id", "provenance", "label_source", "data_source", "source",
)
SCENE_ALIASES = (
    "scene_id", "s2_scene_id", "image_id", "system_index",
    "system:index", "product_id",
)
SAMPLE_ALIASES = (
    "sample_id", "event_id", "observation_id", "patch_id", "filename", "id",
)
TIME_ALIASES = (
    "acquisition_time_utc", "datetime_utc", "acquisition_datetime",
    "scene_time_utc", "timestamp_utc", "date_time",
)
PATH_ALIASES = (
    "resolved_patch_path", "patch_path", "relative_path", "file_path",
    "filepath", "tif_path", "image_path", "filename",
)
EMISSION_ALIASES = (
    "release_rate_kg_h", "emission_kg_hr", "emission_kg_h",
    "emission_rate_kg_h", "matched_positive_release_rate_kg_h",
)

METADATA_NAMES = {
    *LABEL_ALIASES, *SITE_ALIASES, *SOURCE_ALIASES, *SCENE_ALIASES,
    *SAMPLE_ALIASES, *TIME_ALIASES, *PATH_ALIASES, *EMISSION_ALIASES,
    "raster_read_ok", "raster_error", "patch_exists_now", "crs", "transform",
    "height", "width", "band_count", "valid_pixel_fraction",
    "background_class", "background_ndvi_class",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a robust multisource Sentinel-2 generalization baseline."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/Users/happydoraaa/methane_release_project"),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional CSV. If omitted, the script auto-discovers one.",
    )
    parser.add_argument(
        "--random-repeats",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run on generated synthetic data and exit.",
    )
    return parser.parse_args()


def first_column(df: pd.DataFrame, aliases: Iterable[str]) -> Optional[str]:
    lower = {str(c).strip().lower(): c for c in df.columns}
    for name in aliases:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def normalize_string(series: pd.Series, fallback: str) -> pd.Series:
    out = series.astype("string").str.strip()
    out = out.replace(
        {"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA}
    )
    return out.fillna(fallback)


def parse_label(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype(str).str.strip().str.lower()
    numeric = numeric.mask(
        text.isin(["positive", "tp", "plume", "release", "yes", "true"]), 1
    )
    numeric = numeric.mask(
        text.isin(["negative", "tn", "no plume", "no_release", "no", "false"]), 0
    )
    return numeric


def infer_source(row: pd.Series) -> str:
    combined = " ".join(
        str(row.get(name, ""))
        for name in (
            "sample_id", "scene_id", "site_id", "patch_path_raw",
            "original_source_text",
        )
    ).lower()

    rules = (
        ("methaneair", "MethaneAIR"),
        ("scientific reports", "2023_Scientific_Reports"),
        ("2023_sr", "2023_Scientific_Reports"),
        ("2024_amt", "2024_AMT"),
        ("amt", "2024_AMT"),
        ("carbon mapper", "CarbonMapper_derived"),
        ("carbonmapper", "CarbonMapper_derived"),
        ("controlled_release", "ControlledRelease"),
        ("controlled release", "ControlledRelease"),
    )
    for token, result in rules:
        if token in combined:
            return result
    return "unknown_source"


def infer_site_from_text(value: object) -> str:
    text = str(value).lower()
    known = (
        ("casa_grande", "Casa_Grande_AZ"),
        ("casa grande", "Casa_Grande_AZ"),
        ("ehrenberg", "Ehrenberg_AZ"),
        ("alberta", "Alberta"),
        ("permian", "Permian"),
    )
    for token, site in known:
        if token in text:
            return site
    return "unknown_site"


def find_input_csv(project_root: Path, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        path = explicit if explicit.is_absolute() else project_root / explicit
        if not path.exists():
            raise SystemExit(f"Input CSV not found: {path}")
        return path

    outputs = project_root / "outputs"
    for name in PREFERRED_INPUTS:
        candidate = outputs / name
        if candidate.exists():
            return candidate

    # Fallback: choose a CSV that looks like a labelled S2 feature/manifest table.
    candidates = sorted(
        outputs.glob("*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    scored: list[tuple[int, Path]] = []
    for path in candidates:
        try:
            sample = pd.read_csv(path, nrows=5)
        except Exception:
            continue
        columns = {str(c).lower() for c in sample.columns}
        score = 0
        if any(alias.lower() in columns for alias in LABEL_ALIASES):
            score += 5
        if any(alias.lower() in columns for alias in SITE_ALIASES):
            score += 3
        if any(alias.lower() in columns for alias in PATH_ALIASES):
            score += 2
        if any("b11" in c or "b12" in c or "swir" in c for c in columns):
            score += 4
        if "s2" in path.name.lower() or "sentinel" in path.name.lower():
            score += 2
        if score >= 7:
            scored.append((score, path))

    if not scored:
        raise SystemExit(
            f"No suitable input CSV found in {outputs}.\n"
            "Expected one of:\n  - "
            + "\n  - ".join(PREFERRED_INPUTS)
        )

    scored.sort(key=lambda item: (item[0], item[1].stat().st_mtime), reverse=True)
    return scored[0][1]


def resolve_path(project_root: Path, value: object) -> Optional[Path]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return None

    raw = Path(text).expanduser()
    guesses = (
        raw,
        project_root / raw,
        project_root / "outputs" / raw,
        project_root / "data" / raw,
        project_root / "patches" / raw,
        project_root / "images" / raw,
        project_root / "downloads" / raw,
        project_root / "outputs" / raw.name,
        project_root / "patches" / raw.name,
        project_root / "images" / raw.name,
    )
    for guess in guesses:
        if guess.exists() and guess.is_file():
            return guess

    # Search only likely folders by filename.
    for folder_name in ("patches", "images", "data", "downloads", "outputs"):
        folder = project_root / folder_name
        if not folder.exists():
            continue
        matches = list(folder.rglob(raw.name))
        if len(matches) == 1:
            return matches[0]
    return None


def safe_median(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else np.nan


def safe_std(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.std(values)) if values.size else np.nan


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denominator = a + b
    out = np.full_like(a, np.nan, dtype="float64")
    valid = np.isfinite(a) & np.isfinite(b) & (np.abs(denominator) > EPS)
    out[valid] = (a[valid] - b[valid]) / denominator[valid]
    return out


def safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype="float64")
    valid = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > EPS)
    out[valid] = a[valid] / b[valid]
    return out


def extract_raster_features(path: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "raster_read_ok": False,
        "raster_error": "",
    }
    if rasterio is None:
        result["raster_error"] = "rasterio_not_installed"
        return result

    try:
        with rasterio.open(path) as src:
            arr = src.read().astype("float64")
            nodata = src.nodata

        if arr.shape[0] < 6:
            raise ValueError(f"expected >=6 bands, found {arr.shape[0]}")
        arr = arr[:6]
        if nodata is not None:
            arr[arr == nodata] = np.nan

        valid = np.all(np.isfinite(arr), axis=0) & np.any(arr != 0, axis=0)
        if not valid.any():
            raise ValueError("no valid non-zero pixels")

        height, width = arr.shape[1], arr.shape[2]
        yy, xx = np.indices((height, width))
        cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
        radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        scale = min(height, width)
        center = valid & (radius <= 0.18 * scale)
        background = valid & (radius >= 0.30 * scale) & (radius <= 0.48 * scale)
        if not center.any() or not background.any():
            raise ValueError("source-centre/background masks contain no valid pixels")

        names = ("b2", "b3", "b4", "b8", "b11", "b12")
        bands = {}
        for index, name in enumerate(names):
            band = arr[index]
            bands[name] = band
            result[f"{name}_median"] = safe_median(band[valid])
            result[f"{name}_std"] = safe_std(band[valid])
            result[f"{name}_center_median"] = safe_median(band[center])
            result[f"{name}_background_median"] = safe_median(band[background])
            result[f"{name}_center_minus_background"] = (
                result[f"{name}_center_median"]
                - result[f"{name}_background_median"]
            )

        b2, b3, b4, b8, b11, b12 = (bands[n] for n in names)
        derived = {
            "ndvi": normalized_difference(b8, b4),
            "swir_nd_b11_b12": normalized_difference(b11, b12),
            "swir_ratio_b12_b11": safe_ratio(b12, b11),
            "swir_difference_b11_minus_b12": b11 - b12,
        }
        for name, values in derived.items():
            result[f"{name}_median"] = safe_median(values[valid])
            result[f"{name}_std"] = safe_std(values[valid])
            result[f"{name}_center_median"] = safe_median(values[center])
            result[f"{name}_background_median"] = safe_median(values[background])
            result[f"{name}_center_minus_background"] = (
                result[f"{name}_center_median"]
                - result[f"{name}_background_median"]
            )

        result["valid_pixel_fraction"] = float(valid.mean())
        result["raster_read_ok"] = True
        return result

    except Exception as exc:
        result["raster_error"] = f"{type(exc).__name__}: {exc}"
        return result


def canonicalize(raw: pd.DataFrame, project_root: Path) -> pd.DataFrame:
    label_col = first_column(raw, LABEL_ALIASES)
    if label_col is None:
        raise SystemExit(
            "No label column found.\nAvailable columns:\n"
            + "\n".join(map(str, raw.columns))
        )

    site_col = first_column(raw, SITE_ALIASES)
    source_col = first_column(raw, SOURCE_ALIASES)
    scene_col = first_column(raw, SCENE_ALIASES)
    sample_col = first_column(raw, SAMPLE_ALIASES)
    time_col = first_column(raw, TIME_ALIASES)
    path_col = first_column(raw, PATH_ALIASES)
    emission_col = first_column(raw, EMISSION_ALIASES)

    df = raw.copy()
    df["label"] = parse_label(df[label_col])
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)

    df["sample_id"] = (
        normalize_string(df[sample_col], "unknown_sample")
        if sample_col
        else pd.Series([f"sample_{i:05d}" for i in range(len(df))], index=df.index)
    )
    df["scene_id"] = (
        normalize_string(df[scene_col], "unknown_scene")
        if scene_col
        else df["sample_id"]
    )
    df["site_id"] = (
        normalize_string(df[site_col], "unknown_site")
        if site_col
        else pd.Series("unknown_site", index=df.index)
    )
    df["original_source_text"] = (
        normalize_string(df[source_col], "")
        if source_col
        else pd.Series("", index=df.index)
    )
    df["source_origin"] = df["original_source_text"]
    unknown_source = df["source_origin"].isin(["", "unknown", "unknown_source"])
    df.loc[unknown_source, "source_origin"] = df.loc[unknown_source].apply(
        infer_source, axis=1
    )

    if path_col:
        df["patch_path_raw"] = df[path_col]
    else:
        df["patch_path_raw"] = ""

    # Infer missing site from identifiers/path.
    unknown_site = df["site_id"].eq("unknown_site")
    if unknown_site.any():
        df.loc[unknown_site, "site_id"] = df.loc[unknown_site].apply(
            lambda row: infer_site_from_text(
                " ".join(
                    [
                        str(row.get("sample_id", "")),
                        str(row.get("scene_id", "")),
                        str(row.get("patch_path_raw", "")),
                    ]
                )
            ),
            axis=1,
        )

    if time_col:
        df["acquisition_time_utc"] = pd.to_datetime(
            df[time_col], errors="coerce", utc=True
        )
    else:
        df["acquisition_time_utc"] = pd.NaT

    if emission_col:
        df["release_rate_kg_h"] = pd.to_numeric(
            df[emission_col], errors="coerce"
        )
    else:
        df["release_rate_kg_h"] = np.nan

    df["resolved_patch_path"] = df["patch_path_raw"].map(
        lambda value: str(path) if (path := resolve_path(project_root, value)) else ""
    )

    # Avoid one image appearing in both train and test.
    unknown_scene = df["scene_id"].eq("unknown_scene")
    df.loc[
        unknown_scene & df["resolved_patch_path"].ne(""),
        "scene_id",
    ] = df.loc[
        unknown_scene & df["resolved_patch_path"].ne(""),
        "resolved_patch_path",
    ].map(lambda value: Path(value).stem)

    return df.reset_index(drop=True)


def existing_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded_lower = {name.lower() for name in METADATA_NAMES}
    result = []
    for column in df.columns:
        if str(column).lower() in excluded_lower:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            non_null = df[column].notna().sum()
            unique = df[column].nunique(dropna=True)
            if non_null >= 3 and unique >= 2:
                result.append(column)
    return sorted(result)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    features = existing_feature_columns(df)
    methane_like = [
        c for c in features
        if any(token in str(c).lower() for token in ("b11", "b12", "swir", "ndvi"))
    ]

    if len(methane_like) >= 3:
        return df.copy(), "used_existing_numeric_features"

    if "resolved_patch_path" not in df or not df["resolved_patch_path"].ne("").any():
        raise SystemExit(
            "The selected CSV does not contain enough usable numeric features, "
            "and no readable raster path could be resolved.\n"
            "Run with an explicit feature CSV using --input, or confirm patch paths."
        )

    rows = []
    for index, row in df.iterrows():
        record = row.to_dict()
        path_text = str(row["resolved_patch_path"]).strip()
        if path_text:
            record.update(extract_raster_features(Path(path_text)))
        else:
            record.update({
                "raster_read_ok": False,
                "raster_error": "path_not_found",
            })
        rows.append(record)
        if (index + 1) % 10 == 0 or index + 1 == len(df):
            print(f"Feature extraction: {index + 1}/{len(df)}", flush=True)

    result = pd.DataFrame(rows)
    readable = result["raster_read_ok"].fillna(False).astype(bool)
    result = result[readable].copy().reset_index(drop=True)
    if result.empty:
        raise SystemExit(
            "No GeoTIFF could be read. Inspect patch_path_raw and resolved_patch_path "
            "in outputs/500_multisource_canonical_table_v2.csv."
        )
    return result, "extracted_features_from_geotiff"


def select_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    numeric = existing_feature_columns(df)

    swir = [
        c for c in numeric
        if any(token in str(c).lower() for token in ("b11", "b12", "swir"))
    ]
    context = [
        c for c in numeric
        if any(
            token in str(c).lower()
            for token in ("b2", "b3", "b4", "b8", "ndvi")
        )
    ]

    if not swir:
        # Generic fallback for a pre-existing feature table with different names.
        swir = numeric

    feature_sets = {
        "swir_only": sorted(set(swir)),
        "swir_plus_context": sorted(set(swir + context)),
    }
    return {name: cols for name, cols in feature_sets.items() if cols}


def make_pipeline(model_name: str, columns: list[str]) -> Pipeline:
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocess = ColumnTransformer(
        [("numeric", numeric_pipe, columns)],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    if model_name == "dummy":
        estimator = DummyClassifier(strategy="prior")
    elif model_name == "logistic":
        estimator = LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            solver="liblinear",
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(model_name)

    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def safe_metrics(y_true, y_pred, probability) -> dict[str, object]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    two_classes = len(np.unique(y_true)) == 2
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y_true, y_pred))
            if two_classes else np.nan
        ),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(y_true, probability))
            if two_classes else np.nan
        ),
        "average_precision": (
            float(average_precision_score(y_true, probability))
            if np.any(np.asarray(y_true) == 1) else np.nan
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_split(
    data: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    evaluation: str,
    fold: str,
    feature_set_name: str,
    columns: list[str],
    model_name: str,
) -> tuple[Optional[dict[str, object]], list[dict[str, object]]]:
    train = data.iloc[train_indices]
    test = data.iloc[test_indices]
    if train.empty or test.empty or train["label"].nunique() < 2:
        return None, []

    model = make_pipeline(model_name, columns)
    model.fit(train[columns], train["label"])
    probability = model.predict_proba(test[columns])[:, 1]
    prediction = (probability >= 0.5).astype(int)

    metrics = {
        "evaluation": evaluation,
        "fold": fold,
        "feature_set": feature_set_name,
        "model": model_name,
        "n_features": len(columns),
        "n_train": len(train),
        "n_test": len(test),
        "train_positive": int((train["label"] == 1).sum()),
        "train_negative": int((train["label"] == 0).sum()),
        "test_positive": int((test["label"] == 1).sum()),
        "test_negative": int((test["label"] == 0).sum()),
        "train_sites": int(train["site_id"].nunique()),
        "test_sites": int(test["site_id"].nunique()),
        "train_sources": int(train["source_origin"].nunique()),
        "test_sources": int(test["source_origin"].nunique()),
    }
    metrics.update(safe_metrics(test["label"].to_numpy(), prediction, probability))

    predictions = []
    for (_, row), pred, prob in zip(test.iterrows(), prediction, probability):
        predictions.append({
            "evaluation": evaluation,
            "fold": fold,
            "feature_set": feature_set_name,
            "model": model_name,
            "sample_id": row["sample_id"],
            "scene_id": row["scene_id"],
            "site_id": row["site_id"],
            "source_origin": row["source_origin"],
            "true_label": int(row["label"]),
            "predicted_label": int(pred),
            "probability_positive": float(prob),
            "release_rate_kg_h": row.get("release_rate_kg_h", np.nan),
            "acquisition_time_utc": row.get("acquisition_time_utc", pd.NaT),
        })
    return metrics, predictions


def run_group_holdout(
    data: pd.DataFrame,
    group_column: str,
    evaluation: str,
    feature_sets: dict[str, list[str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metrics_rows = []
    prediction_rows = []
    for held_out in sorted(data[group_column].astype(str).unique()):
        mask = data[group_column].astype(str).eq(held_out).to_numpy()
        train_indices = np.flatnonzero(~mask)
        test_indices = np.flatnonzero(mask)
        for fs_name, columns in feature_sets.items():
            for model_name in ("dummy", "logistic"):
                metrics, predictions = evaluate_split(
                    data, train_indices, test_indices, evaluation, held_out,
                    fs_name, columns, model_name,
                )
                if metrics is not None:
                    metrics_rows.append(metrics)
                    prediction_rows.extend(predictions)
    return metrics_rows, prediction_rows


def run_grouped_random(
    data: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    repeats: int,
    test_size: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metrics_rows = []
    prediction_rows = []
    for repeat in range(repeats):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=RANDOM_STATE + repeat,
        )
        train_indices, test_indices = next(
            splitter.split(data, data["label"], groups=data["scene_id"])
        )
        for fs_name, columns in feature_sets.items():
            for model_name in ("dummy", "logistic"):
                metrics, predictions = evaluate_split(
                    data, train_indices, test_indices,
                    "grouped_random_scene", f"repeat_{repeat:02d}",
                    fs_name, columns, model_name,
                )
                if metrics is not None:
                    metrics_rows.append(metrics)
                    prediction_rows.extend(predictions)
    return metrics_rows, prediction_rows


def write_outputs(
    project_root: Path,
    input_path: Path,
    canonical: pd.DataFrame,
    prepared: pd.DataFrame,
    preparation_mode: str,
    feature_sets: dict[str, list[str]],
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    outputs = project_root / "outputs"
    figures = project_root / "figures"
    outputs.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    canonical_path = outputs / "500_multisource_canonical_table_v2.csv"
    features_path = outputs / "501_multisource_features_v2.csv"
    site_source_path = outputs / "502_multisource_site_source_summary_v2.csv"
    metrics_path = outputs / "503_multisource_fold_metrics_v2.csv"
    predictions_path = outputs / "504_multisource_predictions_v2.csv"
    summary_path = outputs / "505_multisource_model_summary_v2.csv"
    report_path = outputs / "506_multisource_model_report_v2.txt"
    figure_path = figures / "507_multisource_loso_balanced_accuracy_v2.png"

    canonical.to_csv(canonical_path, index=False)
    prepared.to_csv(features_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    site_source = (
        prepared.groupby(["site_id", "source_origin"], dropna=False)
        .agg(
            rows=("sample_id", "size"),
            positive=("label", lambda s: int((s == 1).sum())),
            negative=("label", lambda s: int((s == 0).sum())),
            scenes=("scene_id", "nunique"),
            min_emission_kg_h=("release_rate_kg_h", "min"),
            median_emission_kg_h=("release_rate_kg_h", "median"),
            max_emission_kg_h=("release_rate_kg_h", "max"),
        )
        .reset_index()
    )
    site_source.to_csv(site_source_path, index=False)

    summary = (
        metrics.groupby(["evaluation", "feature_set", "model"], dropna=False)
        .agg(
            folds=("fold", "nunique"),
            mean_accuracy=("accuracy", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_recall=("recall", "mean"),
            mean_specificity=("specificity", "mean"),
            mean_f1=("f1", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_average_precision=("average_precision", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)

    plot = metrics[
        (metrics["evaluation"] == "leave_one_site_out")
        & (metrics["model"] == "logistic")
        & (metrics["feature_set"] == "swir_plus_context")
    ].copy()
    if not plot.empty:
        plot = plot.sort_values("balanced_accuracy")
        fig, ax = plt.subplots(figsize=(10, max(4, 0.55 * len(plot))))
        ax.barh(plot["fold"], plot["balanced_accuracy"])
        ax.axvline(0.5, linestyle="--", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Balanced accuracy")
        ax.set_ylabel("Held-out site")
        ax.set_title("Sentinel-2 leave-one-site-out baseline")
        fig.tight_layout()
        fig.savefig(figure_path, dpi=180)
        plt.close(fig)

    report = [
        "=" * 105,
        "MULTISOURCE SENTINEL-2 MODEL REPORT V2",
        "=" * 105,
        "",
        f"Input CSV: {input_path}",
        f"Preparation mode: {preparation_mode}",
        f"Rows in canonical table: {len(canonical)}",
        f"Rows used by model: {len(prepared)}",
        f"Positive rows: {int((prepared['label'] == 1).sum())}",
        f"Negative rows: {int((prepared['label'] == 0).sum())}",
        f"Unique sites: {prepared['site_id'].nunique()}",
        f"Unique sources: {prepared['source_origin'].nunique()}",
        f"Unique scenes: {prepared['scene_id'].nunique()}",
        "",
        "FEATURE SETS",
        "-" * 105,
    ]
    for name, cols in feature_sets.items():
        report.append(f"{name}: {len(cols)} features")

    report.extend([
        "",
        "SITE × SOURCE COUNTS",
        "-" * 105,
        site_source.to_string(index=False),
        "",
        "MODEL SUMMARY",
        "-" * 105,
        summary.to_string(index=False),
        "",
        "HOW TO INTERPRET",
        "-" * 105,
        "1. leave_one_site_out is the main unseen-site generalization result.",
        "2. leave_one_source_out answers whether one controlled-release source/campaign",
        "   transfers to another source/campaign.",
        "3. grouped_random_scene is only an optimistic reference.",
        "4. If SWIR+context greatly outperforms SWIR-only, the model may rely on",
        "   vegetation/soil/site background rather than methane-specific information.",
        "5. A held-out group containing only one class has no valid balanced accuracy/AUROC.",
        "",
        "OUTPUT FILES",
        "-" * 105,
        str(canonical_path),
        str(features_path),
        str(site_source_path),
        str(metrics_path),
        str(predictions_path),
        str(summary_path),
        str(report_path),
        str(figure_path),
    ])
    report_path.write_text("\n".join(report), encoding="utf-8")

    print("\nCreated:")
    for path in (
        canonical_path, features_path, site_source_path, metrics_path,
        predictions_path, summary_path, report_path, figure_path
    ):
        if path.exists():
            print(f"  {path}")


def synthetic_data() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    sources = ["Source_A", "Source_B"]
    sites = ["Site_1", "Site_2", "Site_3", "Site_4", "Site_5"]
    for site_index, site in enumerate(sites):
        source = sources[site_index % 2]
        for i in range(20):
            label = i % 2
            scene = f"{site}_scene_{i // 2:02d}"
            methane_signal = 0.8 * label + rng.normal(0, 0.5)
            site_background = site_index * 0.2 + rng.normal(0, 0.2)
            rows.append({
                "sample_id": f"{site}_{i:03d}",
                "scene_id": scene,
                "site_id": site,
                "source_origin": source,
                "label": label,
                "release_rate_kg_h": 100 + 500 * label + rng.uniform(0, 300),
                "b11_median": methane_signal + site_background,
                "b12_median": -methane_signal + site_background,
                "swir_ratio_b12_b11": methane_signal + rng.normal(0, 0.2),
                "ndvi_median": site_background + rng.normal(0, 0.1),
            })
    return pd.DataFrame(rows)


def run_model(data: pd.DataFrame, repeats: int, test_size: float):
    feature_sets = select_feature_sets(data)
    all_metrics = []
    all_predictions = []

    if data["site_id"].nunique() >= 2:
        metrics, predictions = run_group_holdout(
            data, "site_id", "leave_one_site_out", feature_sets
        )
        all_metrics.extend(metrics)
        all_predictions.extend(predictions)

    if data["source_origin"].nunique() >= 2:
        metrics, predictions = run_group_holdout(
            data, "source_origin", "leave_one_source_out", feature_sets
        )
        all_metrics.extend(metrics)
        all_predictions.extend(predictions)

    if data["scene_id"].nunique() >= 2:
        metrics, predictions = run_grouped_random(
            data, feature_sets, repeats, test_size
        )
        all_metrics.extend(metrics)
        all_predictions.extend(predictions)

    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.DataFrame(all_predictions)
    if metrics_df.empty:
        raise SystemExit(
            "No valid model fold was produced. Check that training folds contain "
            "both positive and negative classes."
        )
    return feature_sets, metrics_df, predictions_df


def main() -> int:
    args = parse_args()

    if args.self_test:
        data = synthetic_data()
        feature_sets, metrics, predictions = run_model(
            data, repeats=3, test_size=0.25
        )
        print("SELF-TEST PASSED")
        print(f"Rows: {len(data)}")
        print(f"Sites: {data['site_id'].nunique()}")
        print(f"Sources: {data['source_origin'].nunique()}")
        print(f"Metric rows: {len(metrics)}")
        print(f"Prediction rows: {len(predictions)}")
        print(f"Feature sets: {feature_sets}")
        return 0

    project_root = args.project_root.expanduser().resolve()
    if not project_root.exists():
        raise SystemExit(f"Project root does not exist: {project_root}")

    input_path = find_input_csv(project_root, args.input)
    print(f"Using input: {input_path}")

    raw = pd.read_csv(input_path)
    if raw.empty:
        raise SystemExit(f"Input CSV is empty: {input_path}")

    canonical = canonicalize(raw, project_root)

    # Save the canonical table early so path problems are inspectable.
    outputs = project_root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    canonical.to_csv(outputs / "500_multisource_canonical_table_v2.csv", index=False)

    prepared, preparation_mode = prepare_features(canonical)

    if prepared["label"].nunique() < 2:
        raise SystemExit("Both positive and negative labels are required.")
    if prepared["site_id"].nunique() < 2:
        raise SystemExit(
            "Fewer than two sites were detected. Inspect site_id in "
            "outputs/500_multisource_canonical_table_v2.csv."
        )
    if prepared["scene_id"].nunique() < 2:
        raise SystemExit("Fewer than two independent scenes were detected.")

    feature_sets, metrics, predictions = run_model(
        prepared,
        repeats=args.random_repeats,
        test_size=args.test_size,
    )

    write_outputs(
        project_root=project_root,
        input_path=input_path,
        canonical=canonical,
        prepared=prepared,
        preparation_mode=preparation_mode,
        feature_sets=feature_sets,
        metrics=metrics,
        predictions=predictions,
    )

    print("\nMain result to open:")
    print(project_root / "outputs/506_multisource_model_report_v2.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
