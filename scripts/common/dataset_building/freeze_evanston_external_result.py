from pathlib import Path

import pandas as pd


METRICS_INPUT = Path(
    "outputs/155_evanston_threshold_calibrated_metrics.csv"
)

PREDICTIONS_INPUT = Path(
    "outputs/154_evanston_threshold_calibrated_predictions.csv"
)

TRAINING_PREDICTIONS_INPUT = Path(
    "outputs/121_landsat_site_calibrated_loso_predictions.csv"
)

OUTPUT = Path(
    "outputs/157_frozen_evanston_external_validation_summary.csv"
)


def main():
    metrics = pd.read_csv(
        METRICS_INPUT,
        low_memory=False,
    )

    predictions = pd.read_csv(
        PREDICTIONS_INPUT,
        low_memory=False,
    )

    training = pd.read_csv(
        TRAINING_PREDICTIONS_INPUT,
        low_memory=False,
    )

    if metrics.empty:
        raise RuntimeError(
            "External metrics table is empty."
        )

    result = metrics.iloc[0]

    evaluation = predictions[
        predictions["external_target"]
        .notna()
    ].copy()

    evaluation["external_target"] = (
        pd.to_numeric(
            evaluation["external_target"],
            errors="coerce",
        )
    )

    evaluation[
        "threshold_calibrated_label"
    ] = pd.to_numeric(
        evaluation[
            "threshold_calibrated_label"
        ],
        errors="coerce",
    )

    primary_training = training.copy()

    if "model_name" in primary_training.columns:
        primary_training = primary_training[
            primary_training["model_name"]
            .astype(str)
            .str.lower()
            .eq("logistic_regression")
        ]

    if "feature_set" in primary_training.columns:
        primary_training = primary_training[
            primary_training["feature_set"]
            .astype(str)
            .str.lower()
            .str.contains(
                "percentile",
                na=False,
            )
        ]

    summary = pd.DataFrame([{
        "validation_site":
            "Evanston",
        "training_sites":
            "Casa Grande | Ehrenberg",
        "model_name":
            result["model_name"],
        "feature_set":
            result["feature_set"],
        "high_emission_threshold_kg_h":
            1000.0,
        "alert_probability_threshold":
            result["threshold"],
        "alert_threshold_source":
            result["threshold_source"],
        "training_oof_rows":
            len(primary_training),
        "external_evaluation_rows":
            len(evaluation),
        "external_high_emission_rows":
            int(
                (
                    evaluation[
                        "external_target"
                    ] == 1
                ).sum()
            ),
        "external_no_release_rows":
            int(
                (
                    evaluation[
                        "external_target"
                    ] == 0
                ).sum()
            ),
        "accuracy":
            result["accuracy"],
        "balanced_accuracy":
            result["balanced_accuracy"],
        "recall_positive":
            result["recall_positive"],
        "recall_negative":
            result["recall_negative"],
        "precision_positive":
            result["precision_positive"],
        "roc_auc":
            result["roc_auc"],
        "true_positive":
            int(result["true_positive"]),
        "true_negative":
            int(result["true_negative"]),
        "false_positive":
            int(result["false_positive"]),
        "false_negative":
            int(result["false_negative"]),
        "result_status":
            "preliminary_third_site_proof_of_concept_success",
        "limitation":
            (
                "Only one Evanston high-emission "
                "controlled-release scene was available."
            ),
    }])

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT,
        index=False,
    )

    print("=" * 100)
    print("FROZEN EVANSTON EXTERNAL RESULT")
    print("=" * 100)

    print(
        summary.to_string(
            index=False,
        )
    )

    print("\nSaved:")
    print(OUTPUT)


if __name__ == "__main__":
    main()
