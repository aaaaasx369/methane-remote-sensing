from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

from train_marss2l_site_disjoint_models import (
    MULTI_FEATURES,
)


DEVELOPMENT_INPUT = Path(
    "outputs/268_marss2l_development_model_ready.csv"
)

VALIDATION_PREDICTION_INPUT = Path(
    "outputs/271_marss2l_validation_predictions.csv"
)

EXTERNAL_INPUT = Path(
    "outputs/280_marss2l_random_forest_external_predictions.csv"
)

SCORE_SUMMARY_OUTPUT = Path(
    "outputs/286_marss2l_validation_external_score_summary.csv"
)

DATASET_METRIC_OUTPUT = Path(
    "outputs/287_marss2l_validation_external_metric_comparison.csv"
)

EXTERNAL_SITE_OUTPUT = Path(
    "outputs/288_marss2l_external_site_error_audit.csv"
)

ERROR_OUTPUT = Path(
    "outputs/289_marss2l_external_wrong_predictions.csv"
)

FEATURE_SHIFT_OUTPUT = Path(
    "outputs/290_marss2l_development_external_feature_shift.csv"
)


FROZEN_THRESHOLD = 0.34


def safe_ratio(numerator, denominator):
    if denominator == 0:
        return np.nan

    return numerator / denominator


def calculate_metrics(frame, dataset_name):
    y_true = (
        frame["target_label"]
        .astype(int)
        .to_numpy()
    )

    score = (
        frame["prediction_score"]
        .astype(float)
        .to_numpy()
    )

    prediction = (
        score >= FROZEN_THRESHOLD
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1],
    ).ravel()

    return {
        "dataset":
            dataset_name,
        "row_count":
            len(frame),
        "site_count":
            frame["site_key"].nunique(),
        "negative_count":
            int((y_true == 0).sum()),
        "positive_count":
            int((y_true == 1).sum()),
        "true_negative":
            int(tn),
        "false_positive":
            int(fp),
        "false_negative":
            int(fn),
        "true_positive":
            int(tp),
        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                prediction,
            ),
        "positive_recall":
            safe_ratio(tp, tp + fn),
        "negative_recall":
            safe_ratio(tn, tn + fp),
        "false_positive_rate":
            safe_ratio(fp, tn + fp),
        "roc_auc":
            roc_auc_score(
                y_true,
                score,
            ),
        "brier_score":
            brier_score_loss(
                y_true,
                score,
            ),
        "mean_prediction_score":
            float(np.mean(score)),
        "threshold":
            FROZEN_THRESHOLD,
    }


def calculate_site_audit(external):
    rows = []

    for site_key, group in external.groupby(
        "site_key"
    ):
        y_true = (
            group["target_label"]
            .astype(int)
            .to_numpy()
        )

        score = (
            group["prediction_score"]
            .astype(float)
            .to_numpy()
        )

        prediction = (
            score >= FROZEN_THRESHOLD
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            prediction,
            labels=[0, 1],
        ).ravel()

        positive_recall = safe_ratio(
            tp,
            tp + fn,
        )

        negative_recall = safe_ratio(
            tn,
            tn + fp,
        )

        rows.append({
            "site_key":
                site_key,
            "row_count":
                len(group),
            "negative_count":
                int((y_true == 0).sum()),
            "positive_count":
                int((y_true == 1).sum()),
            "true_negative":
                int(tn),
            "false_positive":
                int(fp),
            "false_negative":
                int(fn),
            "true_positive":
                int(tp),
            "positive_recall":
                positive_recall,
            "negative_recall":
                negative_recall,
            "false_positive_rate":
                safe_ratio(
                    fp,
                    tn + fp,
                ),
            "balanced_accuracy":
                (
                    (
                        positive_recall
                        + negative_recall
                    ) / 2
                    if (
                        np.isfinite(
                            positive_recall
                        )
                        and np.isfinite(
                            negative_recall
                        )
                    )
                    else np.nan
                ),
            "mean_negative_score":
                group.loc[
                    group["target_label"].eq(0),
                    "prediction_score",
                ].mean(),
            "mean_positive_score":
                group.loc[
                    group["target_label"].eq(1),
                    "prediction_score",
                ].mean(),
            "maximum_negative_score":
                group.loc[
                    group["target_label"].eq(0),
                    "prediction_score",
                ].max(),
            "minimum_positive_score":
                group.loc[
                    group["target_label"].eq(1),
                    "prediction_score",
                ].min(),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "balanced_accuracy",
                "false_positive_rate",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )


def calculate_feature_shift(
    development,
    external,
):
    rows = []

    for feature in MULTI_FEATURES:
        development_values = pd.to_numeric(
            development[feature],
            errors="coerce",
        )

        external_values = pd.to_numeric(
            external[feature],
            errors="coerce",
        )

        development_clean = (
            development_values
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        external_clean = (
            external_values
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        development_median = (
            development_clean.median()
        )

        external_median = (
            external_clean.median()
        )

        development_q25 = (
            development_clean.quantile(0.25)
        )

        development_q75 = (
            development_clean.quantile(0.75)
        )

        development_iqr = (
            development_q75
            - development_q25
        )

        if (
            np.isfinite(development_iqr)
            and abs(development_iqr) > 1e-12
        ):
            robust_median_shift = (
                abs(
                    external_median
                    - development_median
                )
                / abs(development_iqr)
            )
        else:
            robust_median_shift = np.nan

        train_p01 = (
            development_clean.quantile(0.01)
        )

        train_p99 = (
            development_clean.quantile(0.99)
        )

        if len(external_clean):
            external_outside_fraction = (
                (
                    external_clean.lt(
                        train_p01
                    )
                    | external_clean.gt(
                        train_p99
                    )
                )
                .mean()
            )
        else:
            external_outside_fraction = (
                np.nan
            )

        rows.append({
            "feature":
                feature,
            "development_count":
                len(development_clean),
            "external_count":
                len(external_clean),
            "development_missing_fraction":
                development_values.isna().mean(),
            "external_missing_fraction":
                external_values.isna().mean(),
            "development_median":
                development_median,
            "external_median":
                external_median,
            "development_iqr":
                development_iqr,
            "robust_median_shift":
                robust_median_shift,
            "external_outside_train_01_99_fraction":
                external_outside_fraction,
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "robust_median_shift",
                "external_outside_train_01_99_fraction",
            ],
            ascending=[
                False,
                False,
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def main():
    for path in [
        DEVELOPMENT_INPUT,
        VALIDATION_PREDICTION_INPUT,
        EXTERNAL_INPUT,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    development = pd.read_csv(
        DEVELOPMENT_INPUT,
        low_memory=False,
    )

    validation_all = pd.read_csv(
        VALIDATION_PREDICTION_INPUT,
        low_memory=False,
    )

    external = pd.read_csv(
        EXTERNAL_INPUT,
        low_memory=False,
    )

    # sensor_is_lc09 是模型使用的衍生欄位，
    # 原始 CSV 只有 sensor_code，因此在此重新建立。
    for frame in [development, external]:
        frame["sensor_is_lc09"] = (
            frame["sensor_code"]
            .astype(str)
            .str.upper()
            .eq("LC09")
            .astype(int)
        )

    validation = validation_all[
        validation_all["model_name"].eq(
            "random_forest"
        )
    ].copy()

    if validation.empty:
        raise RuntimeError(
            "Random Forest validation "
            "predictions were not found."
        )

    score_frames = []

    for dataset_name, frame in [
        ("validation", validation),
        ("external", external),
    ]:
        for label, group in frame.groupby(
            "target_label"
        ):
            score_frames.append({
                "dataset":
                    dataset_name,
                "target_label":
                    int(label),
                "count":
                    len(group),
                "score_mean":
                    group[
                        "prediction_score"
                    ].mean(),
                "score_std":
                    group[
                        "prediction_score"
                    ].std(),
                "score_p10":
                    group[
                        "prediction_score"
                    ].quantile(0.10),
                "score_p25":
                    group[
                        "prediction_score"
                    ].quantile(0.25),
                "score_median":
                    group[
                        "prediction_score"
                    ].median(),
                "score_p75":
                    group[
                        "prediction_score"
                    ].quantile(0.75),
                "score_p90":
                    group[
                        "prediction_score"
                    ].quantile(0.90),
                "fraction_above_threshold":
                    group[
                        "prediction_score"
                    ].ge(
                        FROZEN_THRESHOLD
                    ).mean(),
            })

    score_summary = pd.DataFrame(
        score_frames
    )

    score_summary.to_csv(
        SCORE_SUMMARY_OUTPUT,
        index=False,
    )

    metrics = pd.DataFrame([
        calculate_metrics(
            validation,
            "validation",
        ),
        calculate_metrics(
            external,
            "external",
        ),
    ])

    metrics.to_csv(
        DATASET_METRIC_OUTPUT,
        index=False,
    )

    site_audit = calculate_site_audit(
        external
    )

    site_audit.to_csv(
        EXTERNAL_SITE_OUTPUT,
        index=False,
    )

    wrong = external[
        external["predicted_label"].ne(
            external["target_label"]
        )
    ].copy()

    useful_columns = [
        "download_id",
        "site_key",
        "sensor_code",
        "external_role",
        "target_label",
        "prediction_score",
        "predicted_label",
        "qa_clear_fraction",
        "ch4_fluxrate",
        "landsat_tile",
        "acquisition_datetime_utc",
        "patch_path",
    ]

    useful_columns = [
        column
        for column in useful_columns
        if column in wrong.columns
    ]

    wrong = wrong[
        useful_columns
    ].sort_values(
        [
            "target_label",
            "prediction_score",
        ],
        ascending=[
            True,
            False,
        ],
    )

    wrong.to_csv(
        ERROR_OUTPUT,
        index=False,
    )

    feature_shift = (
        calculate_feature_shift(
            development,
            external,
        )
    )

    feature_shift.to_csv(
        FEATURE_SHIFT_OUTPUT,
        index=False,
    )

    print("=" * 112)
    print("VALIDATION TO EXTERNAL TRANSFER DIAGNOSIS")
    print("=" * 112)

    print("\nMetrics:")
    print(
        metrics.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    print("\nScore distributions:")
    print(
        score_summary.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    print("\nWorst 15 external sites:")
    print(
        site_audit.head(15).to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    print("\nLargest feature shifts:")
    print(
        feature_shift.head(15).to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    print(
        "\nWrong external predictions:",
        len(wrong),
    )

    print("\nSaved:")
    print(SCORE_SUMMARY_OUTPUT)
    print(DATASET_METRIC_OUTPUT)
    print(EXTERNAL_SITE_OUTPUT)
    print(ERROR_OUTPUT)
    print(FEATURE_SHIFT_OUTPUT)


if __name__ == "__main__":
    main()
