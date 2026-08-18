from pathlib import Path
import pandas as pd


INPUT = Path(
    "outputs/125_stanford_all_release_summaries.csv"
)

OUTPUT = Path(
    "outputs/129_evanston_landsat_release_candidates.csv"
)

SUMMARY_OUTPUT = Path(
    "outputs/130_evanston_landsat_release_summary.csv"
)


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    required = [
        "site_key",
        "release_ID",
        "datetime_utc",
        "lat",
        "lon",
        "ch4_kgh_mean",
        "instrument_code",
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

    df["instrument_code"] = (
        df["instrument_code"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["ch4_kgh_mean"] = pd.to_numeric(
        df["ch4_kgh_mean"],
        errors="coerce",
    )

    df["lat"] = pd.to_numeric(
        df["lat"],
        errors="coerce",
    )

    df["lon"] = pd.to_numeric(
        df["lon"],
        errors="coerce",
    )

    df["datetime_utc"] = pd.to_datetime(
        df["datetime_utc"],
        errors="coerce",
        utc=True,
    )

    candidates = df[
        (df["site_key"] == "evanston")
        & (
            df["instrument_code"]
            .isin(["LS8", "LS9"])
        )
        & df["datetime_utc"].notna()
        & df["lat"].notna()
        & df["lon"].notna()
        & df["ch4_kgh_mean"].notna()
    ].copy()

    candidates["landsat_sensor"] = (
        candidates[
            "instrument_code"
        ].map({
            "LS8": "Landsat-8",
            "LS9": "Landsat-9",
        })
    )

    candidates["acquisition_date"] = (
        candidates["datetime_utc"]
        .dt.strftime("%Y-%m-%d")
    )

    candidates["acquisition_time_utc"] = (
        candidates["datetime_utc"]
        .dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    )

    candidates = (
        candidates
        .sort_values([
            "datetime_utc",
            "instrument_code",
            "ch4_kgh_mean",
        ])
        .drop_duplicates(
            subset=[
                "release_ID",
                "datetime_utc",
                "lat",
                "lon",
                "ch4_kgh_mean",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    candidates[
        "candidate_id"
    ] = [
        f"EV_LS_{number:03d}"
        for number in range(
            1,
            len(candidates) + 1,
        )
    ]

    preferred_columns = [
        "candidate_id",
        "release_ID",
        "site_key",
        "landsat_sensor",
        "instrument_code",
        "acquisition_date",
        "acquisition_time_utc",
        "lat",
        "lon",
        "ch4_kgh_mean",
        "ch4_kgh_sigma",
        "ci95_lower",
        "ci95_upper",
        "PredInt95_lower",
        "PredInt95_upper",
        "source_phase",
        "source_summary_file",
    ]

    output_columns = [
        column
        for column in preferred_columns
        if column in candidates.columns
    ]

    candidates[
        output_columns
    ].to_csv(
        OUTPUT,
        index=False,
    )

    summary = (
        candidates.groupby(
            "landsat_sensor"
        )
        .agg(
            release_rows=(
                "release_ID",
                "size",
            ),
            unique_dates=(
                "acquisition_date",
                "nunique",
            ),
            first_date=(
                "datetime_utc",
                "min",
            ),
            last_date=(
                "datetime_utc",
                "max",
            ),
            flow_min_kg_h=(
                "ch4_kgh_mean",
                "min",
            ),
            flow_median_kg_h=(
                "ch4_kgh_mean",
                "median",
            ),
            flow_mean_kg_h=(
                "ch4_kgh_mean",
                "mean",
            ),
            flow_max_kg_h=(
                "ch4_kgh_mean",
                "max",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print("=" * 100)
    print("EVANSTON LANDSAT RELEASE CANDIDATES")
    print("=" * 100)

    print("\nCandidate rows:", len(candidates))
    print(
        "Unique dates:",
        candidates[
            "acquisition_date"
        ].nunique(),
    )

    print("\nSensor counts:")
    print(
        candidates[
            "landsat_sensor"
        ].value_counts()
    )

    print("\nFlow-rate summary:")
    print(
        candidates[
            "ch4_kgh_mean"
        ].describe(
            percentiles=[
                0.25,
                0.50,
                0.75,
                0.90,
            ]
        )
    )

    print("\nCandidates by date and sensor:")
    print(
        candidates.groupby([
            "acquisition_date",
            "landsat_sensor",
        ])
        .agg(
            releases=(
                "release_ID",
                "size",
            ),
            flow_min_kg_h=(
                "ch4_kgh_mean",
                "min",
            ),
            flow_max_kg_h=(
                "ch4_kgh_mean",
                "max",
            ),
        )
        .reset_index()
        .to_string(
            index=False,
            float_format=lambda value:
                f"{value:.2f}",
        )
    )

    print("\nTop 20 highest releases:")
    print(
        candidates[
            [
                "candidate_id",
                "release_ID",
                "acquisition_time_utc",
                "landsat_sensor",
                "ch4_kgh_mean",
            ]
        ]
        .sort_values(
            "ch4_kgh_mean",
            ascending=False,
        )
        .head(20)
        .to_string(
            index=False,
            float_format=lambda value:
                f"{value:.2f}",
        )
    )

    print("\nSaved:")
    print(OUTPUT)
    print(SUMMARY_OUTPUT)


if __name__ == "__main__":
    main()
