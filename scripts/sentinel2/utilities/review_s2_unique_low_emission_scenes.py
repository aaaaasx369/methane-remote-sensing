from pathlib import Path
import pandas as pd


INTERVAL_INPUT = Path(
    "outputs/310_low_emission_release_intervals_for_s2.csv"
)

OVERLAP_INPUT = Path(
    "outputs/312_s2_low_emission_exact_overlap_summary.csv"
)

DETAIL_OUTPUT = Path(
    "outputs/314_s2_exact_low_emission_provenance.csv"
)

SCENE_OUTPUT = Path(
    "outputs/315_s2_unique_low_emission_scene_groups.csv"
)

REPORT_OUTPUT = Path(
    "outputs/316_s2_unique_low_emission_scene_review.txt"
)


def join_unique(values):
    cleaned = []

    for value in values:
        if pd.isna(value):
            continue

        text = str(value).strip()

        if text and text not in cleaned:
            cleaned.append(text)

    return " | ".join(cleaned)


def join_numbers(values):
    numbers = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    return " | ".join(
        f"{value:.6f}"
        for value in numbers
    )


def main():
    intervals = pd.read_csv(
        INTERVAL_INPUT,
        low_memory=False,
    )

    overlaps = pd.read_csv(
        OVERLAP_INPUT,
        low_memory=False,
    )

    exact_flag = (
        overlaps["has_exact_s2_overlap"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    exact = overlaps[
        exact_flag
    ].copy()

    provenance_columns = [
        column
        for column in [
            "release_interval_id",
            "raw_row_index",
            "source_file",
            "source_sheet",
            "paper_guess",
            "interval_schema",
            "release_rate_source",
        ]
        if column in intervals.columns
    ]

    provenance = intervals[
        provenance_columns
    ].copy()

    detail = exact.merge(
        provenance,
        on="release_interval_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_inventory"),
    )

    detail[
        "best_acquisition_time_utc"
    ] = pd.to_datetime(
        detail[
            "best_acquisition_time_utc"
        ],
        errors="coerce",
        utc=True,
    )

    detail[
        "release_start_utc"
    ] = pd.to_datetime(
        detail[
            "release_start_utc"
        ],
        errors="coerce",
        utc=True,
    )

    detail[
        "release_end_utc"
    ] = pd.to_datetime(
        detail[
            "release_end_utc"
        ],
        errors="coerce",
        utc=True,
    )

    detail[
        "release_rate_kg_h"
    ] = pd.to_numeric(
        detail[
            "release_rate_kg_h"
        ],
        errors="coerce",
    )

    # 同一場址、同一張 S2 scene 視為一個影像樣本。
    group_columns = [
        "site",
        "best_scene_id",
        "best_acquisition_time_utc",
    ]

    scene_rows = []

    for keys, group in detail.groupby(
        group_columns,
        dropna=False,
    ):
        site, scene_id, acquisition = keys

        rates = (
            group["release_rate_kg_h"]
            .dropna()
        )

        raw_rows = (
            group["raw_row_index"]
            .dropna()
            .astype(str)
            .nunique()
            if "raw_row_index" in group.columns
            else 0
        )

        interval_count = len(group)

        if interval_count == 1:
            interpretation = (
                "single_interval"
            )
        elif raw_rows <= 1:
            interpretation = (
                "likely_duplicate_representation"
            )
        else:
            interpretation = (
                "overlapping_intervals_manual_review"
            )

        scene_rows.append({
            "site":
                site,
            "best_scene_id":
                scene_id,
            "best_acquisition_time_utc":
                acquisition,
            "interval_count":
                interval_count,
            "unique_raw_row_count":
                raw_rows,
            "release_interval_ids":
                join_unique(
                    group[
                        "release_interval_id"
                    ]
                ),
            "release_rates_kg_h":
                join_numbers(
                    group[
                        "release_rate_kg_h"
                    ]
                ),
            "minimum_rate_kg_h":
                rates.min()
                if not rates.empty
                else pd.NA,
            "maximum_rate_kg_h":
                rates.max()
                if not rates.empty
                else pd.NA,
            "sum_if_independent_sources_kg_h":
                rates.sum()
                if not rates.empty
                else pd.NA,
            "release_start_times":
                join_unique(
                    group[
                        "release_start_utc"
                    ]
                ),
            "release_end_times":
                join_unique(
                    group[
                        "release_end_utc"
                    ]
                ),
            "rate_sources":
                join_unique(
                    group[
                        "release_rate_source"
                    ]
                )
                if "release_rate_source"
                in group.columns
                else "",
            "source_files":
                join_unique(
                    group["source_file"]
                )
                if "source_file"
                in group.columns
                else "",
            "source_sheets":
                join_unique(
                    group["source_sheet"]
                )
                if "source_sheet"
                in group.columns
                else "",
            "raw_row_indices":
                join_unique(
                    group["raw_row_index"]
                )
                if "raw_row_index"
                in group.columns
                else "",
            "scene_cloud_percentage":
                group[
                    "best_cloudy_pixel_percentage"
                ].min(),
            "review_interpretation":
                interpretation,
            "extreme_low_rate_flag":
                bool(
                    not rates.empty
                    and rates.min() < 10
                ),
            "manual_decision":
                "",
            "final_release_rate_kg_h":
                pd.NA,
            "review_notes":
                "",
        })

    scenes = pd.DataFrame(
        scene_rows
    ).sort_values(
        "best_acquisition_time_utc"
    ).reset_index(drop=True)

    detail.to_csv(
        DETAIL_OUTPUT,
        index=False,
    )

    scenes.to_csv(
        SCENE_OUTPUT,
        index=False,
    )

    with REPORT_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as report:
        report.write(
            "=" * 110
            + "\n"
        )
        report.write(
            "UNIQUE SENTINEL-2 LOW-EMISSION SCENE REVIEW\n"
        )
        report.write(
            "=" * 110
            + "\n"
        )

        report.write(
            f"\nExact interval matches: {len(detail)}\n"
        )

        report.write(
            f"Unique S2 scenes: {len(scenes)}\n"
        )

        for _, row in scenes.iterrows():
            report.write(
                "\n"
                + "#" * 110
                + "\n"
            )

            report.write(
                f"Site: {row['site']}\n"
            )

            report.write(
                "Acquisition: "
                f"{row['best_acquisition_time_utc']}\n"
            )

            report.write(
                f"Scene: {row['best_scene_id']}\n"
            )

            report.write(
                "Interval count: "
                f"{row['interval_count']}\n"
            )

            report.write(
                "Raw rows: "
                f"{row['raw_row_indices']}\n"
            )

            report.write(
                "Release rates: "
                f"{row['release_rates_kg_h']} kg/h\n"
            )

            report.write(
                "Sum if independent sources: "
                f"{row['sum_if_independent_sources_kg_h']} kg/h\n"
            )

            report.write(
                "Rate sources: "
                f"{row['rate_sources']}\n"
            )

            report.write(
                "Release starts: "
                f"{row['release_start_times']}\n"
            )

            report.write(
                "Release ends: "
                f"{row['release_end_times']}\n"
            )

            report.write(
                "Source files: "
                f"{row['source_files']}\n"
            )

            report.write(
                "Source sheets: "
                f"{row['source_sheets']}\n"
            )

            report.write(
                "Scene cloud percentage: "
                f"{row['scene_cloud_percentage']}\n"
            )

            report.write(
                "Automatic interpretation: "
                f"{row['review_interpretation']}\n"
            )

            report.write(
                "Extreme low-rate flag: "
                f"{row['extreme_low_rate_flag']}\n"
            )

    print("=" * 100)
    print("UNIQUE SENTINEL-2 LOW-EMISSION SCENES")
    print("=" * 100)

    print(
        "\nExact interval matches:",
        len(detail),
    )

    print(
        "Unique Sentinel-2 scenes:",
        len(scenes),
    )

    print("\nScene groups:")
    print(
        scenes[
            [
                "site",
                "best_acquisition_time_utc",
                "interval_count",
                "unique_raw_row_count",
                "release_rates_kg_h",
                "sum_if_independent_sources_kg_h",
                "rate_sources",
                "review_interpretation",
                "extreme_low_rate_flag",
            ]
        ].to_string(
            index=False,
        )
    )

    print("\nSaved:")
    print(DETAIL_OUTPUT)
    print(SCENE_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
