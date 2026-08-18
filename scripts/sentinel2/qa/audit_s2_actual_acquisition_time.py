from pathlib import Path

import numpy as np
import pandas as pd


S2_INPUT = Path(
    "outputs/22_controlled_release_s2_dataset_table.csv"
)

EVENT_INPUT = Path(
    "outputs/12_final_availability_with_flags.csv"
)

OUTPUT = Path(
    "outputs/293_s2_actual_acquisition_time_audit.csv"
)


def main():
    s2 = pd.read_csv(
        S2_INPUT,
        low_memory=False,
    )

    events = pd.read_csv(
        EVENT_INPUT,
        low_memory=False,
    )

    for frame in [s2, events]:
        frame["event_id"] = (
            frame["event_id"]
            .astype(str)
            .str.strip()
        )

    # 避免 merge 後分不清楚哪一個 datetime。
    events = events.rename(columns={
        "datetime_utc":
            "original_event_time",
        "s2_first_time":
            "availability_s2_time",
    })

    keep_columns = [
        "event_id",
        "original_event_time",
        "availability_s2_time",
        "satellite_from_paper",
        "true_release",
        "emission_tph_mean",
        "emission_tph_median",
        "emission_tph_max",
        "source_files_str",
    ]

    keep_columns = [
        column
        for column in keep_columns
        if column in events.columns
    ]

    if events["event_id"].duplicated().any():
        raise RuntimeError(
            "Availability table event_id is not unique."
        )

    merged = s2.merge(
        events[keep_columns],
        on="event_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    # 原始事件時間，例如 PRISMA/GHGSat overpass。
    merged["original_event_time"] = (
        pd.to_datetime(
            merged["original_event_time"],
            errors="coerce",
            utc=True,
        )
    )

    # Patch table 記錄的真正 Sentinel-2 scene time。
    merged["s2_image_time"] = (
        pd.to_datetime(
            merged["s2_image_time"],
            errors="coerce",
            utc=True,
        )
    )

    # Availability 表中的 Sentinel-2 scene time，作為備援。
    if "availability_s2_time" in merged.columns:
        merged["availability_s2_time"] = (
            pd.to_datetime(
                merged["availability_s2_time"],
                errors="coerce",
                utc=True,
            )
        )
    else:
        merged["availability_s2_time"] = pd.NaT

    merged["actual_s2_time"] = (
        merged["s2_image_time"]
        .fillna(
            merged["availability_s2_time"]
        )
    )

    merged["s2_event_time_difference_minutes"] = (
        (
            merged["actual_s2_time"]
            - merged["original_event_time"]
        )
        .abs()
        .dt.total_seconds()
        / 60.0
    )

    delta = merged[
        "s2_event_time_difference_minutes"
    ]

    merged["time_match_category"] = np.select(
        [
            delta.le(15),
            delta.le(60),
            delta.le(180),
            delta.le(24 * 60),
        ],
        [
            "within_15_minutes",
            "15_to_60_minutes",
            "1_to_3_hours",
            "3_to_24_hours",
        ],
        default="over_24_hours_or_missing",
    )

    merged["candidate_within_15_minutes"] = (
        delta.le(15)
    )

    merged["candidate_within_60_minutes"] = (
        delta.le(60)
    )

    for statistic in [
        "mean",
        "median",
        "max",
    ]:
        tph_column = (
            f"emission_tph_{statistic}"
        )

        if tph_column in merged.columns:
            merged[
                f"emission_kg_h_{statistic}"
            ] = (
                pd.to_numeric(
                    merged[tph_column],
                    errors="coerce",
                )
                * 1000.0
            )

    if (
        "emission_kg_h_mean"
        in merged.columns
    ):
        merged["emission_bin"] = pd.cut(
            merged["emission_kg_h_mean"],
            bins=[
                -np.inf,
                0,
                200,
                500,
                1000,
                2000,
                np.inf,
            ],
            labels=[
                "0_negative",
                "0_to_200",
                "200_to_500",
                "500_to_1000",
                "1000_to_2000",
                "2000_plus",
            ],
            include_lowest=True,
        )

    merged.to_csv(
        OUTPUT,
        index=False,
    )

    print("=" * 100)
    print("SENTINEL-2 ACTUAL ACQUISITION TIME AUDIT")
    print("=" * 100)

    print("\nTotal patch rows:", len(merged))

    print("\nEvent merge status:")
    print(
        merged["_merge"]
        .value_counts(dropna=False)
    )

    print("\nActual S2 time available:")
    print(
        merged["actual_s2_time"]
        .notna()
        .value_counts()
    )

    print("\nTime-match categories:")
    print(
        merged["time_match_category"]
        .value_counts(dropna=False)
    )

    print("\nTime match by current label:")
    print(
        pd.crosstab(
            merged["label"],
            merged["time_match_category"],
            margins=True,
        )
    )

    print("\nCurrent positive patches within 15 minutes:")
    print(
        (
            merged["label"].eq(1)
            & merged[
                "candidate_within_15_minutes"
            ]
        ).sum()
    )

    print("\nCurrent positive patches within 60 minutes:")
    print(
        (
            merged["label"].eq(1)
            & merged[
                "candidate_within_60_minutes"
            ]
        ).sum()
    )

    if "emission_bin" in merged.columns:
        print(
            "\nEmission bins for positive patches "
            "within 60 minutes:"
        )

        subset = merged[
            merged["label"].eq(1)
            & merged[
                "candidate_within_60_minutes"
            ]
        ]

        print(
            subset["emission_bin"]
            .value_counts(
                dropna=False
            )
            .sort_index()
        )

    print("\nSaved:")
    print(OUTPUT)

    print(
        "\n注意：within 15/60 minutes "
        "目前只是候選時間匹配，"
        "仍要取得真正 release start/end "
        "才能稱為 strict positive。"
    )


if __name__ == "__main__":
    main()
