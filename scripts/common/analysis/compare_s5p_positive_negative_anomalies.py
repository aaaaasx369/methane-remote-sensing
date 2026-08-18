from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score


INPUT = Path(
    "outputs/507_s5p_regional_anomaly_valid_features_v1.csv"
)

PRIMARY_OUTPUT = Path(
    "outputs/509_s5p_primary_labeled_features_v1.csv"
)

ORBIT_OUTPUT = Path(
    "outputs/510_s5p_primary_orbit_group_audit_v1.csv"
)

COMPARISON_OUTPUT = Path(
    "outputs/511_s5p_positive_negative_feature_comparison_v1.csv"
)

REPORT_OUTPUT = Path(
    "outputs/512_s5p_positive_negative_comparison_report_v1.txt"
)


FEATURE_COLUMNS = [
    "source_minus_background_mean_ppb",
    "source_minus_background_median_ppb",
    "source_minus_background_percent",
    "mean_anomaly_z_score",
    "median_anomaly_robust_z_score",
]

PRIMARY_ROLE = (
    "primary_near_time_regional_context"
)


def benjamini_hochberg(p_values):
    """Return Benjamini-Hochberg adjusted p-values."""

    values = np.asarray(
        p_values,
        dtype=float,
    )

    adjusted = np.full(
        len(values),
        np.nan,
    )

    valid_indices = np.where(
        np.isfinite(values)
    )[0]

    if len(valid_indices) == 0:
        return adjusted

    valid_p = values[
        valid_indices
    ]

    order = np.argsort(valid_p)
    ranked = valid_p[order]

    number = len(ranked)

    ranked_adjusted = (
        ranked
        * number
        / np.arange(1, number + 1)
    )

    ranked_adjusted = np.minimum.accumulate(
        ranked_adjusted[::-1]
    )[::-1]

    ranked_adjusted = np.clip(
        ranked_adjusted,
        0,
        1,
    )

    restored = np.empty(number)

    restored[order] = ranked_adjusted

    adjusted[
        valid_indices
    ] = restored

    return adjusted


def cliffs_delta(positive, negative):
    """
    Positive delta means positive events tend to have
    larger feature values than negative events.
    """

    positive = np.asarray(
        positive,
        dtype=float,
    )

    negative = np.asarray(
        negative,
        dtype=float,
    )

    positive = positive[
        np.isfinite(positive)
    ]

    negative = negative[
        np.isfinite(negative)
    ]

    if (
        len(positive) == 0
        or len(negative) == 0
    ):
        return np.nan

    greater = 0
    smaller = 0

    for value in positive:
        greater += np.sum(
            value > negative
        )

        smaller += np.sum(
            value < negative
        )

    return (
        greater - smaller
    ) / (
        len(positive)
        * len(negative)
    )


def effect_size_label(delta):
    if pd.isna(delta):
        return "missing"

    magnitude = abs(delta)

    if magnitude < 0.147:
        return "negligible"

    if magnitude < 0.330:
        return "small"

    if magnitude < 0.474:
        return "medium"

    return "large"


def feature_decision(
    q_value,
    delta,
    directional_auc,
):
    if (
        pd.notna(q_value)
        and q_value < 0.05
        and pd.notna(delta)
        and abs(delta) >= 0.33
        and pd.notna(directional_auc)
        and directional_auc >= 0.65
    ):
        return (
            "supports_exploratory_classifier"
        )

    if (
        (
            pd.notna(q_value)
            and q_value < 0.10
        )
        or (
            pd.notna(delta)
            and abs(delta) >= 0.20
        )
        or (
            pd.notna(directional_auc)
            and directional_auc >= 0.60
        )
    ):
        return (
            "weak_or_uncertain_separation"
        )

    return (
        "no_useful_separation_detected"
    )


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    frame = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    required = [
        "event_id",
        "s5p_system_index",
        "s5p_true_release",
        "recommended_analysis_role",
        "time_difference_hours",
    ]

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise KeyError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    feature_columns = [
        column
        for column in FEATURE_COLUMNS
        if column in frame.columns
    ]

    if not feature_columns:
        raise KeyError(
            "No expected S5P anomaly features found."
        )

    frame["s5p_true_release"] = pd.to_numeric(
        frame["s5p_true_release"],
        errors="coerce",
    )

    frame["time_difference_hours"] = pd.to_numeric(
        frame["time_difference_hours"],
        errors="coerce",
    )

    for column in feature_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    primary = frame[
        frame[
            "recommended_analysis_role"
        ].eq(PRIMARY_ROLE)
        & frame[
            "s5p_true_release"
        ].isin([0.0, 1.0])
    ].copy()

    primary = primary.dropna(
        subset=feature_columns,
        how="all",
    )

    primary.to_csv(
        PRIMARY_OUTPUT,
        index=False,
    )

    # --------------------------------------------------
    # Orbit grouping audit
    # --------------------------------------------------
    orbit_records = []

    for orbit_id, group in primary.groupby(
        "s5p_system_index",
        dropna=False,
        sort=True,
    ):
        positive_count = int(
            group[
                "s5p_true_release"
            ].eq(1).sum()
        )

        negative_count = int(
            group[
                "s5p_true_release"
            ].eq(0).sum()
        )

        orbit_records.append({
            "s5p_system_index":
                orbit_id,

            "event_count":
                len(group),

            "positive_event_count":
                positive_count,

            "negative_event_count":
                negative_count,

            "mixed_release_labels":
                (
                    positive_count > 0
                    and negative_count > 0
                ),

            "event_ids":
                " | ".join(
                    group["event_id"]
                    .astype(str)
                    .tolist()
                ),

            "minimum_time_difference_hours":
                group[
                    "time_difference_hours"
                ].min(),

            "median_time_difference_hours":
                group[
                    "time_difference_hours"
                ].median(),

            "maximum_time_difference_hours":
                group[
                    "time_difference_hours"
                ].max(),
        })

    orbit_audit = pd.DataFrame(
        orbit_records
    )

    orbit_audit.to_csv(
        ORBIT_OUTPUT,
        index=False,
    )

    # --------------------------------------------------
    # Positive-negative feature comparisons
    # --------------------------------------------------
    comparison_records = []

    labels = primary[
        "s5p_true_release"
    ].astype(int)

    for feature in feature_columns:
        positive = primary.loc[
            labels.eq(1),
            feature,
        ].dropna()

        negative = primary.loc[
            labels.eq(0),
            feature,
        ].dropna()

        if (
            len(positive) > 0
            and len(negative) > 0
        ):
            test = mannwhitneyu(
                positive,
                negative,
                alternative="two-sided",
            )

            p_value = float(
                test.pvalue
            )

            u_statistic = float(
                test.statistic
            )

            delta = cliffs_delta(
                positive,
                negative,
            )

            combined = pd.concat(
                [positive, negative]
            )

            combined_labels = np.concatenate([
                np.ones(len(positive)),
                np.zeros(len(negative)),
            ])

            raw_auc = roc_auc_score(
                combined_labels,
                combined.to_numpy(),
            )

            directional_auc = max(
                raw_auc,
                1.0 - raw_auc,
            )

        else:
            p_value = np.nan
            u_statistic = np.nan
            delta = np.nan
            raw_auc = np.nan
            directional_auc = np.nan

        comparison_records.append({
            "feature":
                feature,

            "positive_count":
                len(positive),

            "negative_count":
                len(negative),

            "positive_mean":
                positive.mean(),

            "negative_mean":
                negative.mean(),

            "mean_difference_positive_minus_negative":
                (
                    positive.mean()
                    - negative.mean()
                ),

            "positive_median":
                positive.median(),

            "negative_median":
                negative.median(),

            "median_difference_positive_minus_negative":
                (
                    positive.median()
                    - negative.median()
                ),

            "positive_std":
                positive.std(),

            "negative_std":
                negative.std(),

            "positive_fraction_above_zero":
                (
                    positive.gt(0).mean()
                    if len(positive)
                    else np.nan
                ),

            "negative_fraction_above_zero":
                (
                    negative.gt(0).mean()
                    if len(negative)
                    else np.nan
                ),

            "mann_whitney_u":
                u_statistic,

            "mann_whitney_p":
                p_value,

            "cliffs_delta":
                delta,

            "effect_size":
                effect_size_label(
                    delta
                ),

            "raw_single_feature_auc":
                raw_auc,

            "direction_independent_auc":
                directional_auc,
        })

    comparison = pd.DataFrame(
        comparison_records
    )

    comparison[
        "mann_whitney_q_bh"
    ] = benjamini_hochberg(
        comparison[
            "mann_whitney_p"
        ].to_numpy()
    )

    comparison[
        "analysis_decision"
    ] = [
        feature_decision(
            row[
                "mann_whitney_q_bh"
            ],
            row[
                "cliffs_delta"
            ],
            row[
                "direction_independent_auc"
            ],
        )
        for _, row in comparison.iterrows()
    ]

    comparison.to_csv(
        COMPARISON_OUTPUT,
        index=False,
    )

    # --------------------------------------------------
    # Summaries
    # --------------------------------------------------
    label_summary = (
        primary[
            "s5p_true_release"
        ]
        .value_counts()
        .reindex(
            [1.0, 0.0],
            fill_value=0,
        )
    )

    total_orbits = int(
        orbit_audit[
            "s5p_system_index"
        ].nunique()
    )

    repeated_orbits = int(
        orbit_audit[
            "event_count"
        ].gt(1).sum()
    )

    mixed_label_orbits = int(
        orbit_audit[
            "mixed_release_labels"
        ].sum()
    )

    strong_features = comparison[
        comparison[
            "analysis_decision"
        ].eq(
            "supports_exploratory_classifier"
        )
    ]

    weak_features = comparison[
        comparison[
            "analysis_decision"
        ].eq(
            "weak_or_uncertain_separation"
        )
    ]

    if not strong_features.empty:
        overall_decision = (
            "proceed_to_grouped_exploratory_"
            "classifier"
        )

    elif not weak_features.empty:
        overall_decision = (
            "only_optional_grouped_sensitivity_"
            "classifier"
        )

    else:
        overall_decision = (
            "do_not_build_s5p_classifier_"
            "from_current_features"
        )

    report_lines = [
        "=" * 115,
        "S5P PRIMARY POSITIVE-NEGATIVE ANOMALY COMPARISON V1",
        "=" * 115,
        "",
        (
            "Input valid regional feature events: "
            f"{len(frame)}"
        ),
        (
            "Primary <=6 h labeled events: "
            f"{len(primary)}"
        ),
        (
            "Primary positive events: "
            f"{int(label_summary.loc[1.0])}"
        ),
        (
            "Primary negative events: "
            f"{int(label_summary.loc[0.0])}"
        ),
        "",
        f"Unique S5P orbit groups: {total_orbits}",
        (
            "Orbit groups containing multiple events: "
            f"{repeated_orbits}"
        ),
        (
            "Orbit groups containing mixed labels: "
            f"{mixed_label_orbits}"
        ),
        "",
        "Feature comparison:",
        comparison.to_string(index=False),
        "",
        f"Overall decision: {overall_decision}",
        "",
        "Interpretation:",
        (
            "All comparisons are exploratory because none "
            "of the S5P observations has confirmed native "
            "pixel acquisition time inside a release interval."
        ),
        (
            "Any later classifier must split data by "
            "s5p_system_index so observations from the same "
            "orbit cannot enter both training and test sets."
        ),
        (
            "A positive-negative difference would indicate "
            "near-time regional association, not confirmed "
            "controlled-release plume detection."
        ),
    ]

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("=" * 115)
    print(
        "S5P POSITIVE-NEGATIVE ANOMALY COMPARISON"
    )
    print("=" * 115)

    print(
        "\nPrimary <=6 h labeled events:",
        len(primary),
    )

    print("\nPrimary event labels:")
    print(label_summary)

    print(
        "\nUnique S5P orbit groups:",
        total_orbits,
    )

    print(
        "Orbit groups containing multiple events:",
        repeated_orbits,
    )

    print(
        "Orbit groups containing mixed labels:",
        mixed_label_orbits,
    )

    print("\nFeature comparison:")
    print(
        comparison[
            [
                "feature",
                "positive_count",
                "negative_count",
                "positive_median",
                "negative_median",
                "median_difference_positive_minus_negative",
                "mann_whitney_p",
                "mann_whitney_q_bh",
                "cliffs_delta",
                "effect_size",
                "raw_single_feature_auc",
                "analysis_decision",
            ]
        ].to_string(index=False)
    )

    print(
        "\nOverall decision:",
        overall_decision,
    )

    print("\nSaved:")
    print(PRIMARY_OUTPUT)
    print(ORBIT_OUTPUT)
    print(COMPARISON_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
