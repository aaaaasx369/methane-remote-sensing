from pathlib import Path

import pandas as pd


CODE_FILES = [
    Path("calibrate_landsat_site_baseline_and_evaluate.py"),
    Path("build_evanston_external_anomaly_features.py"),
    Path("evaluate_evanston_external_frozen_model.py"),
    Path("calibrate_frozen_alert_threshold.py"),
]

FEATURE_INPUT = Path(
    "outputs/118_landsat_site_calibrated_anomaly_features.csv"
)

LOSO_INPUT = Path(
    "outputs/121_landsat_site_calibrated_loso_predictions.csv"
)

EVANSTON_FEATURE_INPUT = Path(
    "outputs/146_evanston_external_temporal_features.csv"
)

OUTPUT = Path(
    "outputs/236_frozen_primary_model_contract.txt"
)

TERMS = [
    "cal_temporal_z_source_p95_percentile",
    "cal_temporal_z_source_p95_z",
    "temporal_z_source_p95",
    "percentile",
    "rank",
    "median",
    "mad",
    "robust_scale",
    "LogisticRegression",
    "StandardScaler",
    "Pipeline",
    "class_weight",
    "solver",
    "max_iter",
    "random_state",
    "predict_proba",
    "fit(",
    "coef_",
    "intercept_",
    "0.559805",
    "actual_label",
    "release_rate_kg_h",
]

CONTEXT_LINES = 8


def merged_windows(hit_lines, total_lines):
    windows = []

    for line_number in hit_lines:
        start = max(
            0,
            line_number - CONTEXT_LINES,
        )

        end = min(
            total_lines,
            line_number + CONTEXT_LINES + 1,
        )

        if (
            windows
            and start <= windows[-1][1]
        ):
            windows[-1] = (
                windows[-1][0],
                max(
                    windows[-1][1],
                    end,
                ),
            )
        else:
            windows.append(
                (start, end)
            )

    return windows


def extract_code_context(path):
    output_lines = []

    output_lines.append(
        "=" * 115
    )
    output_lines.append(
        f"CODE FILE: {path}"
    )
    output_lines.append(
        "=" * 115
    )

    if not path.exists():
        output_lines.append(
            "FILE NOT FOUND"
        )
        return output_lines

    lines = path.read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines()

    hit_lines = []

    for index, line in enumerate(lines):
        if any(
            term.lower() in line.lower()
            for term in TERMS
        ):
            hit_lines.append(index)

    if not hit_lines:
        output_lines.append(
            "No matching terms found."
        )
        return output_lines

    windows = merged_windows(
        hit_lines,
        len(lines),
    )

    for start, end in windows:
        output_lines.append("")

        for index in range(start, end):
            output_lines.append(
                f"{index + 1:5d}: "
                f"{lines[index]}"
            )

    return output_lines


def inspect_development_features():
    output_lines = []

    output_lines.append("")
    output_lines.append(
        "=" * 115
    )
    output_lines.append(
        "DEVELOPMENT FEATURE TABLE"
    )
    output_lines.append(
        "=" * 115
    )

    if not FEATURE_INPUT.exists():
        output_lines.append(
            f"FILE NOT FOUND: {FEATURE_INPUT}"
        )
        return output_lines

    df = pd.read_csv(
        FEATURE_INPUT,
        low_memory=False,
    )

    output_lines.append(
        f"Rows: {len(df)}"
    )

    output_lines.append(
        f"Columns: {len(df.columns)}"
    )

    relevant_columns = [
        column
        for column in df.columns
        if any(
            keyword in column.lower()
            for keyword in [
                "scene",
                "site",
                "label",
                "release",
                "source_p95",
                "temporal",
                "cal_",
                "exclude",
                "valid",
                "role",
            ]
        )
    ]

    output_lines.append("")
    output_lines.append(
        "Relevant columns:"
    )

    output_lines.extend(
        relevant_columns
    )

    if "site_key" in df.columns:
        output_lines.append("")
        output_lines.append(
            "Rows by site:"
        )

        output_lines.append(
            df["site_key"]
            .value_counts(dropna=False)
            .to_string()
        )

    label_column = None

    for candidate in [
        "actual_label",
        "label",
        "evaluation_label",
    ]:
        if candidate in df.columns:
            label_column = candidate
            break

    if label_column is not None:
        output_lines.append("")
        output_lines.append(
            f"Labels using {label_column}:"
        )

        output_lines.append(
            df[label_column]
            .value_counts(dropna=False)
            .sort_index()
            .to_string()
        )

    display_columns = [
        column
        for column in [
            "scene_key",
            "site_key",
            "release_rate_kg_h",
            label_column,
            "temporal_z_source_p95",
            "cal_temporal_z_source_p95_z",
            "cal_temporal_z_source_p95_percentile",
        ]
        if (
            column is not None
            and column in df.columns
        )
    ]

    output_lines.append("")
    output_lines.append(
        "Feature examples:"
    )

    output_lines.append(
        df[
            display_columns
        ].head(20).to_string(
            index=False,
            max_colwidth=100,
        )
    )

    return output_lines


def inspect_primary_oof():
    output_lines = []

    output_lines.append("")
    output_lines.append(
        "=" * 115
    )
    output_lines.append(
        "PRIMARY OOF PREDICTIONS"
    )
    output_lines.append(
        "=" * 115
    )

    if not LOSO_INPUT.exists():
        output_lines.append(
            f"FILE NOT FOUND: {LOSO_INPUT}"
        )
        return output_lines

    df = pd.read_csv(
        LOSO_INPUT,
        low_memory=False,
    )

    primary = df[
        df["model_name"].eq(
            "logistic_regression"
        )
        & df["feature_set"].eq(
            "calibrated_source_p95_percentile_1"
        )
    ].copy()

    output_lines.append(
        f"Primary OOF rows: {len(primary)}"
    )

    output_lines.append("")
    output_lines.append(
        "Labels:"
    )

    output_lines.append(
        primary["actual_label"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    output_lines.append("")
    output_lines.append(
        "Rows by held-out site:"
    )

    output_lines.append(
        primary["test_site"]
        .value_counts()
        .to_string()
    )

    output_lines.append("")
    output_lines.append(
        "Primary OOF rows:"
    )

    columns = [
        "scene_key",
        "site_key",
        "release_rate_kg_h",
        "train_site",
        "test_site",
        "actual_label",
        "prediction_score",
        "predicted_label",
        "correct",
    ]

    output_lines.append(
        primary[
            [
                column
                for column in columns
                if column in primary.columns
            ]
        ].to_string(
            index=False,
            max_colwidth=100,
        )
    )

    return output_lines


def inspect_evanston_features():
    output_lines = []

    output_lines.append("")
    output_lines.append(
        "=" * 115
    )
    output_lines.append(
        "EVANSTON FEATURE CONTRACT"
    )
    output_lines.append(
        "=" * 115
    )

    if not EVANSTON_FEATURE_INPUT.exists():
        output_lines.append(
            f"FILE NOT FOUND: "
            f"{EVANSTON_FEATURE_INPUT}"
        )
        return output_lines

    df = pd.read_csv(
        EVANSTON_FEATURE_INPUT,
        low_memory=False,
    )

    output_lines.append(
        f"Rows: {len(df)}"
    )

    relevant_columns = [
        column
        for column in df.columns
        if any(
            keyword in column.lower()
            for keyword in [
                "role",
                "label",
                "source_p95",
                "temporal",
                "cal_",
                "patch",
                "scene",
            ]
        )
    ]

    output_lines.append("")
    output_lines.append(
        "Relevant columns:"
    )

    output_lines.extend(
        relevant_columns
    )

    display_columns = [
        column
        for column in [
            "external_role",
            "evaluation_label",
            "temporal_z_source_p95",
            "cal_temporal_z_source_p95_z",
            "cal_temporal_z_source_p95_percentile",
            "prediction_score",
        ]
        if column in df.columns
    ]

    output_lines.append("")
    output_lines.append(
        "Evanston rows:"
    )

    output_lines.append(
        df[
            display_columns
        ].to_string(
            index=False,
            max_colwidth=100,
        )
    )

    return output_lines


def main():
    report_lines = []

    for path in CODE_FILES:
        report_lines.extend(
            extract_code_context(path)
        )

        report_lines.append("")

    report_lines.extend(
        inspect_development_features()
    )

    report_lines.extend(
        inspect_primary_oof()
    )

    report_lines.extend(
        inspect_evanston_features()
    )

    report = "\n".join(
        report_lines
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        report,
        encoding="utf-8",
    )

    print(report)

    print("\nSaved:")
    print(OUTPUT)


if __name__ == "__main__":
    main()
