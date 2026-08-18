from pathlib import Path

import pandas as pd


CLEAN_DATASET = Path(
    "outputs/43_landsat_unique_clean_features.csv"
)

PREDICTIONS = Path(
    "outputs/45_landsat_loocv_predictions.csv"
)

FULL_OUTPUT = Path(
    "outputs/48_landsat_context5_prediction_review.csv"
)

ERROR_OUTPUT = Path(
    "outputs/49_landsat_context5_wrong_predictions.csv"
)


MODEL_NAME = "logistic_context_5"


def main():
    if not CLEAN_DATASET.exists():
        raise FileNotFoundError(
            f"Missing file: {CLEAN_DATASET}"
        )

    if not PREDICTIONS.exists():
        raise FileNotFoundError(
            f"Missing file: {PREDICTIONS}"
        )

    clean_df = pd.read_csv(CLEAN_DATASET)
    prediction_df = pd.read_csv(PREDICTIONS)

    model_predictions = prediction_df[
        prediction_df["model"] == MODEL_NAME
    ].copy()

    if len(model_predictions) == 0:
        raise ValueError(
            f"No predictions found for {MODEL_NAME}"
        )

    required_columns = [
        "raster_group_id",
        "true_label",
        "predicted_label",
        "probability_label_1",
        "correct",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in model_predictions.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing prediction columns: "
            + ", ".join(missing_columns)
        )

    prediction_columns = [
        "raster_group_id",
        "true_label",
        "predicted_label",
        "probability_label_1",
        "correct",
    ]

    model_predictions = model_predictions[
        prediction_columns
    ].copy()

    review_df = clean_df.merge(
        model_predictions,
        on="raster_group_id",
        how="inner",
        validate="one_to_one",
    )

    review_df["probability_margin_from_0_5"] = (
        review_df["probability_label_1"] - 0.5
    )

    review_df["absolute_probability_margin"] = (
        review_df[
            "probability_margin_from_0_5"
        ].abs()
    )

    review_df["prediction_type"] = "correct"

    review_df.loc[
        (
            (review_df["true_label"] == 0)
            & (review_df["predicted_label"] == 1)
        ),
        "prediction_type",
    ] = "false_positive"

    review_df.loc[
        (
            (review_df["true_label"] == 1)
            & (review_df["predicted_label"] == 0)
        ),
        "prediction_type",
    ] = "false_negative"

    review_df = review_df.sort_values(
        by="probability_label_1",
        ascending=False,
    ).reset_index(drop=True)

    wrong_df = review_df[
        review_df["prediction_type"] != "correct"
    ].copy()

    FULL_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    review_df.to_csv(
        FULL_OUTPUT,
        index=False,
    )

    wrong_df.to_csv(
        ERROR_OUTPUT,
        index=False,
    )

    print("=" * 90)
    print("LANDSAT CONTEXT-5 ERROR ANALYSIS")
    print("=" * 90)

    print(f"\nTotal predictions: {len(review_df)}")

    print("\nPrediction-type counts:")
    print(
        review_df["prediction_type"]
        .value_counts()
    )

    print("\nProbability summary by true label:")
    print(
        review_df.groupby("true_label")[
            "probability_label_1"
        ].describe()
    )

    preferred_columns = [
        "raster_group_id",
        "true_label",
        "predicted_label",
        "probability_label_1",
        "prediction_type",
        "event_id",
        "site_name",
        "landsat_sensor",
        "landsat_image_time",
        "datetime_utc",
        "representative_time_difference_seconds",
        "emission_tph_mean",
        "emission_tph_median",
        "emission_tph_max",
        "label_decision_source",
        "label_decision_confidence",
        "duplicate_source_row_count",
    ]

    display_columns = [
        column
        for column in preferred_columns
        if column in wrong_df.columns
    ]

    print("\nWrong predictions:")
    if len(wrong_df) == 0:
        print("None")
    else:
        print(
            wrong_df[display_columns]
            .to_string(index=False)
        )

    print("\nAll predictions ordered by probability:")
    all_display_columns = [
        column
        for column in preferred_columns
        if column in review_df.columns
    ]

    print(
        review_df[all_display_columns]
        .to_string(index=False)
    )

    print("\nSaved:")
    print(FULL_OUTPUT)
    print(ERROR_OUTPUT)


if __name__ == "__main__":
    main()
