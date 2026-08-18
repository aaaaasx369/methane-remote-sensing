from pathlib import Path

import numpy as np
import pandas as pd


INPUT_CSV = Path(
    "outputs/35_landsat_patch_features.csv"
)

FEATURE_MANIFEST_CSV = Path(
    "outputs/36_landsat_candidate_feature_columns.csv"
)

AUDIT_TEXT = Path(
    "outputs/36_landsat_modeling_readiness_audit.txt"
)


# Only columns created from the satellite image are eligible.
# This prevents filename, event metadata, emission rate, and label leakage.
FEATURE_PREFIXES = (
    "blue_",
    "green_",
    "red_",
    "nir_",
    "swir1_",
    "swir2_",
    "ndvi_",
    "ndmi_",
    "nbr_",
    "ndsi_",
    "swir_normalized_difference_",
    "swir2_over_swir1_",
    "swir1_over_swir2_",
    "log_swir1_over_swir2_",
)


def write_line(lines, text=""):
    print(text)
    lines.append(str(text))


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)

    lines = []

    write_line(lines, "=" * 80)
    write_line(lines, "LANDSAT MODELING READINESS AUDIT")
    write_line(lines, "=" * 80)

    write_line(lines, f"\nInput file: {INPUT_CSV}")
    write_line(lines, f"Rows: {len(df)}")
    write_line(lines, f"Total columns: {len(df.columns)}")

    # --------------------------------------------------------
    # Label checks
    # --------------------------------------------------------

    if "label" not in df.columns:
        raise ValueError("The dataset has no label column.")

    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce",
    )

    write_line(lines, "\nLabel counts:")
    write_line(
        lines,
        df["label"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string(),
    )

    # --------------------------------------------------------
    # Sensor balance
    # --------------------------------------------------------

    if "landsat_sensor" in df.columns:
        write_line(lines, "\nSensor counts:")
        write_line(
            lines,
            df["landsat_sensor"]
            .value_counts(dropna=False)
            .to_string(),
        )

        write_line(lines, "\nLabel by sensor:")
        write_line(
            lines,
            pd.crosstab(
                df["landsat_sensor"],
                df["label"],
                margins=True,
            ).to_string(),
        )

    # --------------------------------------------------------
    # Select image-derived features only
    # --------------------------------------------------------

    candidate_feature_columns = [
        column
        for column in df.columns
        if column.startswith(FEATURE_PREFIXES)
        and pd.api.types.is_numeric_dtype(df[column])
    ]

    write_line(
        lines,
        f"\nCandidate image feature columns: "
        f"{len(candidate_feature_columns)}",
    )

    feature_manifest = pd.DataFrame({
        "feature_column": candidate_feature_columns,
    })

    feature_manifest.to_csv(
        FEATURE_MANIFEST_CSV,
        index=False,
    )

    if not candidate_feature_columns:
        raise ValueError(
            "No candidate image feature columns were found."
        )

    X = (
        df[candidate_feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )

    # --------------------------------------------------------
    # Missing and infinite-value checks
    # --------------------------------------------------------

    missing_counts = X.isna().sum()
    maximum_missing = int(missing_counts.max())

    write_line(
        lines,
        f"\nMaximum missing count in one feature: "
        f"{maximum_missing}",
    )

    features_with_missing = missing_counts[
        missing_counts > 0
    ].sort_values(ascending=False)

    write_line(
        lines,
        f"Features containing missing values: "
        f"{len(features_with_missing)}",
    )

    if len(features_with_missing) > 0:
        write_line(
            lines,
            features_with_missing.to_string(),
        )

    # --------------------------------------------------------
    # Constant features
    # --------------------------------------------------------

    unique_counts = X.nunique(dropna=True)

    constant_features = unique_counts[
        unique_counts <= 1
    ].index.tolist()

    write_line(
        lines,
        f"\nConstant feature columns: "
        f"{len(constant_features)}",
    )

    if constant_features:
        for column in constant_features:
            write_line(lines, f"  {column}")

    # --------------------------------------------------------
    # Duplicate feature vectors
    # --------------------------------------------------------

    duplicate_mask = X.duplicated(
        keep=False
    )

    write_line(
        lines,
        f"\nRows belonging to duplicated feature vectors: "
        f"{int(duplicate_mask.sum())}",
    )

    if duplicate_mask.any():
        display_columns = [
            column
            for column in [
                "event_id",
                "patch_id",
                "filename",
                "file_name",
                "resolved_patch_path",
                "label",
                "landsat_sensor",
            ]
            if column in df.columns
        ]

        write_line(
            lines,
            df.loc[
                duplicate_mask,
                display_columns,
            ].to_string(index=False),
        )

    # --------------------------------------------------------
    # Find possible event/site grouping columns
    # --------------------------------------------------------

    group_keywords = (
        "event",
        "site",
        "release_id",
        "experiment",
        "source_id",
        "flight",
    )

    possible_group_columns = [
        column
        for column in df.columns
        if any(
            keyword in column.lower()
            for keyword in group_keywords
        )
    ]

    write_line(
        lines,
        "\nPossible event/site grouping columns:",
    )

    if not possible_group_columns:
        write_line(lines, "  None found")
    else:
        for column in possible_group_columns:
            non_null = df[column].dropna()
            unique_count = non_null.nunique()

            repeated_rows = int(
                non_null.duplicated(
                    keep=False
                ).sum()
            )

            write_line(
                lines,
                f"  {column}: "
                f"non-null={len(non_null)}, "
                f"unique={unique_count}, "
                f"rows_in_repeated_groups={repeated_rows}",
            )

            # Check whether the same group contains both labels
            if len(non_null) > 0:
                temporary = df.loc[
                    df[column].notna(),
                    [column, "label"],
                ].copy()

                label_counts_per_group = (
                    temporary.groupby(column)["label"]
                    .nunique()
                )

                mixed_label_groups = int(
                    (label_counts_per_group > 1).sum()
                )

                write_line(
                    lines,
                    f"      groups containing both labels: "
                    f"{mixed_label_groups}",
                )

    # --------------------------------------------------------
    # Reflectance diagnostics
    # --------------------------------------------------------

    diagnostic_columns = [
        column
        for column in [
            "raw_dn_min",
            "raw_dn_max",
            "reflectance_min",
            "reflectance_max",
            "reflectance_fraction_below_0",
            "reflectance_fraction_above_1",
            "valid_pixel_fraction",
        ]
        if column in df.columns
    ]

    if diagnostic_columns:
        write_line(
            lines,
            "\nReflectance and validity diagnostics:",
        )

        write_line(
            lines,
            df[diagnostic_columns]
            .describe()
            .transpose()
            .to_string(),
        )

    # --------------------------------------------------------
    # Potential leakage-related columns
    # --------------------------------------------------------

    leakage_keywords = (
        "label",
        "release_rate",
        "emission",
        "flow_rate",
        "plume",
        "detected",
        "download_status",
        "filename",
        "file_name",
        "path",
    )

    potential_leakage_columns = [
        column
        for column in df.columns
        if any(
            keyword in column.lower()
            for keyword in leakage_keywords
        )
        and column not in candidate_feature_columns
    ]

    write_line(
        lines,
        "\nColumns that must NOT automatically enter the model:",
    )

    for column in potential_leakage_columns:
        write_line(lines, f"  {column}")

    # --------------------------------------------------------
    # Save audit
    # --------------------------------------------------------

    AUDIT_TEXT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    write_line(lines, "\nSaved:")
    write_line(lines, str(FEATURE_MANIFEST_CSV))
    write_line(lines, str(AUDIT_TEXT))


if __name__ == "__main__":
    main()
