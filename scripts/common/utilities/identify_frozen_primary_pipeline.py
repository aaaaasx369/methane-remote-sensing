from pathlib import Path

import pandas as pd


PREDICTION_INPUT = Path(
    "outputs/121_landsat_site_calibrated_loso_predictions.csv"
)

GROUP_SUMMARY_OUTPUT = Path(
    "outputs/236_frozen_pipeline_group_summary.csv"
)

PRIMARY_CANDIDATE_OUTPUT = Path(
    "outputs/237_frozen_primary_oof_candidates.csv"
)

SEARCH_TERMS = [
    "cal_temporal_z_source_p95_percentile",
    "calibrated_source_p95_percentile",
    "temporal_z_source_p95",
    "source_p95_percentile",
    "prediction_score",
    "site_calibrated",
    "0.559805",
]


def search_python_files():
    matches = []

    for path in Path(".").rglob("*.py"):
        if any(
            part in {
                ".venv",
                ".git",
                "__pycache__",
            }
            for part in path.parts
        ):
            continue

        try:
            text = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            continue

        found_terms = [
            term
            for term in SEARCH_TERMS
            if term in text
        ]

        if found_terms:
            matches.append({
                "path": str(path),
                "terms": ", ".join(found_terms),
            })

    return pd.DataFrame(matches)


def search_csv_headers():
    matches = []

    search_directories = [
        Path("outputs"),
        Path("raw_data"),
    ]

    for directory in search_directories:
        if not directory.exists():
            continue

        for path in directory.rglob("*.csv"):
            try:
                preview = pd.read_csv(
                    path,
                    nrows=3,
                    low_memory=False,
                )
            except Exception:
                continue

            columns = list(preview.columns)

            matched_columns = [
                column
                for column in columns
                if any(
                    term.lower()
                    in column.lower()
                    for term in SEARCH_TERMS
                )
            ]

            useful_structure = (
                "scene_key" in columns
                and "site_key" in columns
            )

            if matched_columns or useful_structure:
                matches.append({
                    "path":
                        str(path),
                    "rows_previewed":
                        len(preview),
                    "matched_columns":
                        ", ".join(
                            matched_columns
                        ),
                    "has_scene_site_keys":
                        useful_structure,
                    "all_columns":
                        ", ".join(columns),
                })

    return pd.DataFrame(matches)


def main():
    if not PREDICTION_INPUT.exists():
        raise FileNotFoundError(
            PREDICTION_INPUT
        )

    df = pd.read_csv(
        PREDICTION_INPUT,
        low_memory=False,
    )

    required = [
        "scene_key",
        "site_key",
        "model_name",
        "feature_set",
        "actual_label",
        "prediction_score",
        "threshold",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise KeyError(
            f"Missing columns: {missing}"
        )

    group_summary = (
        df.groupby(
            [
                "model_name",
                "feature_set",
            ],
            dropna=False,
        )
        .agg(
            row_count=(
                "scene_key",
                "size",
            ),
            unique_scene_count=(
                "scene_key",
                "nunique",
            ),
            unique_site_count=(
                "site_key",
                "nunique",
            ),
            negative_count=(
                "actual_label",
                lambda values:
                    int((values == 0).sum()),
            ),
            positive_count=(
                "actual_label",
                lambda values:
                    int((values == 1).sum()),
            ),
            prediction_score_min=(
                "prediction_score",
                "min",
            ),
            prediction_score_median=(
                "prediction_score",
                "median",
            ),
            prediction_score_max=(
                "prediction_score",
                "max",
            ),
            non_null_threshold_count=(
                "threshold",
                "count",
            ),
            threshold_min=(
                "threshold",
                "min",
            ),
            threshold_max=(
                "threshold",
                "max",
            ),
        )
        .reset_index()
    )

    group_summary[
        "looks_like_primary"
    ] = (
        group_summary[
            "row_count"
        ].eq(16)
        & (
            group_summary[
                "model_name"
            ]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.contains(
                "logistic|calibrat",
                regex=True,
            )
            |
            group_summary[
                "feature_set"
            ]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.contains(
                "percentile|calibrat|p95",
                regex=True,
            )
            |
            group_summary[
                "non_null_threshold_count"
            ].eq(16)
        )
    )

    group_summary = (
        group_summary.sort_values(
            [
                "looks_like_primary",
                "row_count",
                "model_name",
                "feature_set",
            ],
            ascending=[
                False,
                True,
                True,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    candidate_groups = group_summary[
        group_summary[
            "looks_like_primary"
        ]
    ][
        [
            "model_name",
            "feature_set",
        ]
    ].copy()

    if not candidate_groups.empty:
        candidate_rows = df.merge(
            candidate_groups,
            on=[
                "model_name",
                "feature_set",
            ],
            how="inner",
            validate="many_to_many",
        )
    else:
        candidate_rows = pd.DataFrame(
            columns=df.columns
        )

    python_matches = search_python_files()
    csv_matches = search_csv_headers()

    GROUP_SUMMARY_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    group_summary.to_csv(
        GROUP_SUMMARY_OUTPUT,
        index=False,
    )

    candidate_rows.to_csv(
        PRIMARY_CANDIDATE_OUTPUT,
        index=False,
    )

    print("=" * 110)
    print("MODEL / FEATURE GROUPS IN OUTPUT 121")
    print("=" * 110)

    print(
        group_summary.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    print("\n" + "=" * 110)
    print("LIKELY PRIMARY GROUPS")
    print("=" * 110)

    if candidate_groups.empty:
        print("No group selected automatically.")
    else:
        print(
            candidate_groups.to_string(
                index=False
            )
        )

    print("\n" + "=" * 110)
    print("PYTHON FILES CONTAINING PIPELINE TERMS")
    print("=" * 110)

    if python_matches.empty:
        print("No matching Python files.")
    else:
        print(
            python_matches.to_string(
                index=False,
                max_colwidth=150,
            )
        )

    print("\n" + "=" * 110)
    print("CSV FILES THAT MAY CONTAIN TRAINING FEATURES")
    print("=" * 110)

    if csv_matches.empty:
        print("No matching CSV files.")
    else:
        display_columns = [
            "path",
            "matched_columns",
            "has_scene_site_keys",
        ]

        print(
            csv_matches[
                display_columns
            ].to_string(
                index=False,
                max_colwidth=180,
            )
        )

    print("\nSaved:")
    print(GROUP_SUMMARY_OUTPUT)
    print(PRIMARY_CANDIDATE_OUTPUT)


if __name__ == "__main__":
    main()
