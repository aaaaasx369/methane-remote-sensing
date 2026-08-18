from pathlib import Path

import pandas as pd


INPUT_CSV = Path(
    "outputs/37_landsat_raster_duplicate_rows.csv"
)

GROUP_REVIEW_CSV = Path(
    "outputs/39_landsat_raster_group_review.csv"
)

MIXED_ROWS_CSV = Path(
    "outputs/40_landsat_mixed_label_rows_for_review.csv"
)

NONCONFLICTING_UNIQUE_CSV = Path(
    "outputs/41_landsat_unique_nonconflicting_features.csv"
)


def join_unique(series):
    """
    Join distinct non-null values for provenance review.
    """
    values = (
        series.dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    return " | ".join(values)


def find_review_columns(df):
    """
    Find columns that may help determine the correct scene-level label.
    """
    keywords = (
        "time",
        "date",
        "start",
        "end",
        "release",
        "acquisition",
        "overpass",
        "emission",
        "flow",
        "site",
        "sensor",
        "cloud",
        "source",
    )

    return [
        column
        for column in df.columns
        if any(
            keyword in column.lower()
            for keyword in keywords
        )
    ]


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)

    required_columns = [
        "raster_group_id",
        "label",
        "pixel_hash",
    ]

    missing_required = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_required:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_required)
        )

    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce",
    )

    print("=" * 90)
    print("LANDSAT MIXED-LABEL GROUP REVIEW")
    print("=" * 90)

    print(f"\nInput rows: {len(df)}")
    print(
        "Unique raster groups:",
        df["raster_group_id"].nunique(),
    )

    review_metadata_columns = find_review_columns(df)

    print("\nPossible time/release metadata columns:")

    if review_metadata_columns:
        for column in review_metadata_columns:
            print(f"  {column}")
    else:
        print("  None found")

    core_columns = [
        column
        for column in [
            "raster_group_id",
            "event_id",
            "filename",
            "resolved_patch_path",
            "label",
            "site_name",
            "landsat_sensor",
        ]
        if column in df.columns
    ]

    group_rows = []
    representative_rows = []
    mixed_group_ids = []

    for raster_group_id, group in df.groupby(
        "raster_group_id",
        sort=False,
    ):
        labels = sorted(
            group["label"]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        mixed_labels = len(labels) > 1

        if mixed_labels:
            mixed_group_ids.append(
                raster_group_id
            )

        group_row = {
            "raster_group_id": raster_group_id,
            "n_rows": len(group),
            "labels": ",".join(
                map(str, labels)
            ),
            "n_unique_labels": len(labels),
            "mixed_labels": mixed_labels,
        }

        for column in core_columns:
            if column in (
                "raster_group_id",
                "label",
            ):
                continue

            group_row[column] = join_unique(
                group[column]
            )

        for column in review_metadata_columns:
            if column in group_row:
                continue

            group_row[column] = join_unique(
                group[column]
            )

        group_rows.append(group_row)

        # Keep one representative only when the raster group
        # has one unambiguous label.
        if not mixed_labels:
            representative = group.iloc[0].copy()

            representative[
                "duplicate_source_row_count"
            ] = len(group)

            if "event_id" in group.columns:
                representative[
                    "duplicate_source_event_ids"
                ] = join_unique(
                    group["event_id"]
                )

            if "filename" in group.columns:
                representative[
                    "duplicate_source_filenames"
                ] = join_unique(
                    group["filename"]
                )

            representative_rows.append(
                representative
            )

    group_df = pd.DataFrame(group_rows)

    group_df = group_df.sort_values(
        by=[
            "mixed_labels",
            "n_rows",
            "raster_group_id",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    )

    mixed_df = df[
        df["raster_group_id"].isin(
            mixed_group_ids
        )
    ].copy()

    mixed_df = mixed_df.sort_values(
        by=[
            "raster_group_id",
            "label",
        ]
    )

    nonconflicting_unique_df = pd.DataFrame(
        representative_rows
    )

    GROUP_REVIEW_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    group_df.to_csv(
        GROUP_REVIEW_CSV,
        index=False,
    )

    mixed_df.to_csv(
        MIXED_ROWS_CSV,
        index=False,
    )

    nonconflicting_unique_df.to_csv(
        NONCONFLICTING_UNIQUE_CSV,
        index=False,
    )

    print("\n" + "=" * 90)
    print("REVIEW SUMMARY")
    print("=" * 90)

    print(
        "\nTotal unique raster groups:",
        len(group_df),
    )

    print(
        "Mixed-label raster groups:",
        int(group_df["mixed_labels"].sum()),
    )

    print(
        "Rows inside mixed-label groups:",
        len(mixed_df),
    )

    print(
        "Unique nonconflicting rasters:",
        len(nonconflicting_unique_df),
    )

    if len(nonconflicting_unique_df) > 0:
        print(
            "\nTemporary nonconflicting "
            "unique-label counts:"
        )

        print(
            nonconflicting_unique_df[
                "label"
            ].value_counts().sort_index()
        )

    print("\nMixed-label group details:")

    display_columns = [
        column
        for column in [
            "raster_group_id",
            "event_id",
            "filename",
            "label",
            "site_name",
            "landsat_sensor",
        ]
        if column in mixed_df.columns
    ]

    # Add useful time/release columns to terminal output.
    for column in review_metadata_columns:
        if (
            column not in display_columns
            and len(display_columns) < 18
        ):
            display_columns.append(column)

    if len(mixed_df) == 0:
        print("No mixed-label groups.")
    else:
        print(
            mixed_df[
                display_columns
            ].to_string(index=False)
        )

    print("\nSaved:")
    print(GROUP_REVIEW_CSV)
    print(MIXED_ROWS_CSV)
    print(NONCONFLICTING_UNIQUE_CSV)


if __name__ == "__main__":
    main()
