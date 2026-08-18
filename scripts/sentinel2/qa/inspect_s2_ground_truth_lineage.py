from pathlib import Path
import pandas as pd


FILES = {
    "s2_patches":
        Path("outputs/22_controlled_release_s2_dataset_table.csv"),

    "availability":
        Path("outputs/12_final_availability_with_flags.csv"),

    "unique_events_09":
        Path("outputs/09_final_unique_overpass_events.csv"),

    "unique_events_07":
        Path("outputs/07_unique_overpass_events.csv"),

    "gee_events":
        Path("outputs/10_final_events_for_gee.csv"),

    "raw_candidate_rows":
        Path("outputs/02_candidate_event_rows.csv"),
}


KEYWORDS = [
    "event",
    "source",
    "file",
    "input",
    "row",
    "index",
    "paper",
    "dataset",
    "site",
    "lat",
    "lon",
    "date",
    "time",
    "start",
    "end",
    "release",
    "emission",
    "flux",
    "rate",
    "kg",
    "tph",
]


def relevant_columns(columns):
    return [
        column
        for column in columns
        if any(
            keyword in column.lower()
            for keyword in KEYWORDS
        )
    ]


def main():
    schemas = {}

    print("=" * 120)
    print("FILE SCHEMAS")
    print("=" * 120)

    for name, path in FILES.items():
        if not path.exists():
            print(f"\nMISSING: {path}")
            continue

        preview = pd.read_csv(
            path,
            nrows=5,
            low_memory=False,
        )

        schemas[name] = list(
            preview.columns
        )

        print("\n" + "-" * 120)
        print(name)
        print(path)
        print("Column count:", len(preview.columns))

        print("\nRelevant columns:")
        for column in relevant_columns(
            preview.columns
        ):
            print(" ", column)

        display_columns = relevant_columns(
            preview.columns
        )[:20]

        if display_columns:
            print("\nFirst rows:")
            print(
                preview[display_columns]
                .head(3)
                .to_string(
                    index=False,
                    max_colwidth=80,
                )
            )

    required = [
        "s2_patches",
        "availability",
        "raw_candidate_rows",
    ]

    if not all(
        name in schemas
        for name in required
    ):
        return

    print("\n" + "=" * 120)
    print("COMMON COLUMNS WITH RAW CANDIDATE TABLE")
    print("=" * 120)

    raw_columns = set(
        schemas["raw_candidate_rows"]
    )

    for name in [
        "availability",
        "unique_events_09",
        "unique_events_07",
        "gee_events",
        "s2_patches",
    ]:
        if name not in schemas:
            continue

        common = sorted(
            set(schemas[name])
            & raw_columns
        )

        print(f"\n{name} ↔ raw_candidate_rows:")
        print(common)

    print("\n" + "=" * 120)
    print("SENTINEL-2 EVENT-ID COVERAGE")
    print("=" * 120)

    s2 = pd.read_csv(
        FILES["s2_patches"],
        low_memory=False,
    )

    availability = pd.read_csv(
        FILES["availability"],
        low_memory=False,
    )

    for frame in [s2, availability]:
        frame["event_id"] = (
            frame["event_id"]
            .astype(str)
            .str.strip()
        )

    joined = s2.merge(
        availability,
        on="event_id",
        how="left",
        suffixes=("_s2", "_event"),
        indicator=True,
    )

    print("S2 patch rows:", len(s2))
    print(
        "Unique S2 event IDs:",
        s2["event_id"].nunique(),
    )

    print("\nJoin status:")
    print(
        joined["_merge"]
        .value_counts(dropna=False)
    )

    print("\nAvailability event ID duplicates:")
    print(
        availability[
            "event_id"
        ].duplicated(
            keep=False
        ).sum()
    )

    useful = [
        column
        for column in [
            "event_id",
            "filename",
            "site_s2",
            "site_event",
            "source_dataset_s2",
            "source_dataset_event",
            "input_csv_s2",
            "input_csv_event",
            "datetime_utc_s2",
            "datetime_utc_event",
            "search_start",
            "search_end",
            "emission_tph_mean",
            "emission_tph_max",
            "true_release",
            "_merge",
        ]
        if column in joined.columns
    ]

    print("\nJoined preview:")
    print(
        joined[useful]
        .head(15)
        .to_string(
            index=False,
            max_colwidth=100,
        )
    )

    output = Path(
        "outputs/292_s2_ground_truth_lineage_join_preview.csv"
    )

    joined.to_csv(
        output,
        index=False,
    )

    print("\nSaved:")
    print(output)


if __name__ == "__main__":
    main()
