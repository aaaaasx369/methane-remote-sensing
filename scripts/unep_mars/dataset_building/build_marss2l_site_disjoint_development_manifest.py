from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


CANDIDATE_INPUT = Path(
    "outputs/251_marss2l_development_candidate_images.csv"
)

EXTERNAL_INDEX_INPUT = Path(
    "outputs/234_marss2l_external_patch_index_v1_2.csv"
)

SITE_SPLIT_OUTPUT = Path(
    "outputs/253_marss2l_site_disjoint_development_sites.csv"
)

MANIFEST_OUTPUT = Path(
    "outputs/254_marss2l_development_download_manifest.csv"
)

SUMMARY_OUTPUT = Path(
    "outputs/255_marss2l_development_manifest_summary.csv"
)


RANDOM_STATE = 42
VALIDATION_FRACTION = 0.20

CALIBRATION_NEGATIVES_PER_SITE = 5
MODEL_NEGATIVES_PER_SITE = 5
MAX_POSITIVES_PER_SITE = 5

HIGH_EMISSION_THRESHOLD_KG_H = 1000.0

SPLIT_VERSION = (
    "marss2l_development_v1_"
    "site_disjoint_80_20"
)


def parse_boolean(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        })
    )


def nearest_positive_days(
    timestamp,
    positive_times,
):
    if pd.isna(timestamp) or not positive_times:
        return np.nan

    return min(
        abs(
            (
                timestamp
                - positive_time
            ).total_seconds()
        )
        for positive_time in positive_times
    ) / 86400.0


def choose_flux_spread(
    positives,
    maximum_count,
):
    """
    不只選最大排放量，讓選出的正樣本涵蓋
    低、中、高排放範圍。
    """
    positives = (
        positives.sort_values(
            [
                "ch4_fluxrate",
                "acquisition_datetime_utc",
                "image_key",
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    if len(positives) <= maximum_count:
        return positives.copy()

    indices = np.linspace(
        0,
        len(positives) - 1,
        maximum_count,
    )

    indices = np.round(
        indices
    ).astype(int)

    indices = np.unique(indices)

    # 避免 round 後少於指定數量。
    if len(indices) < maximum_count:
        remaining = [
            index
            for index in range(len(positives))
            if index not in set(indices)
        ]

        indices = np.concatenate([
            indices,
            remaining[
                :maximum_count - len(indices)
            ],
        ])

    indices = sorted(
        indices[:maximum_count]
    )

    return positives.iloc[
        indices
    ].copy()


def main():
    for path in [
        CANDIDATE_INPUT,
        EXTERNAL_INDEX_INPUT,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    images = pd.read_csv(
        CANDIDATE_INPUT,
        low_memory=False,
    )

    external = pd.read_csv(
        EXTERNAL_INDEX_INPUT,
        low_memory=False,
    )

    required = [
        "site_key",
        "image_key",
        "benchmark_role",
        "satellite_normalized",
        "acquisition_datetime_utc",
        "ch4_fluxrate",
        "eligible_site",
    ]

    missing = [
        column
        for column in required
        if column not in images.columns
    ]

    if missing:
        raise KeyError(
            f"Missing columns: {missing}"
        )

    images["site_key"] = (
        images["site_key"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    images["eligible_site_parsed"] = (
        parse_boolean(
            images["eligible_site"]
        )
    )

    images[
        "acquisition_datetime_utc"
    ] = pd.to_datetime(
        images[
            "acquisition_datetime_utc"
        ],
        errors="coerce",
        utc=True,
    )

    images["ch4_fluxrate"] = pd.to_numeric(
        images["ch4_fluxrate"],
        errors="coerce",
    )

    images["satellite_normalized"] = (
        images["satellite_normalized"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    images = images[
        images["eligible_site_parsed"].eq(True)
        & images["site_key"].ne("")
        & images["benchmark_role"].isin([
            "high_emission_positive",
            "no_plume_negative",
        ])
    ].copy()

    # 同一影像可能同時出現在官方 train/val；
    # 合併後只保留一次。
    images = (
        images.sort_values(
            [
                "site_key",
                "benchmark_role",
                "ch4_fluxrate",
            ],
            ascending=[
                True,
                True,
                False,
            ],
            na_position="last",
        )
        .drop_duplicates(
            subset=[
                "site_key",
                "image_key",
                "benchmark_role",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    site_summary = (
        images.groupby("site_key")
        .agg(
            positive_count=(
                "benchmark_role",
                lambda values:
                    int(
                        (
                            values
                            == "high_emission_positive"
                        ).sum()
                    ),
            ),
            negative_count=(
                "benchmark_role",
                lambda values:
                    int(
                        (
                            values
                            == "no_plume_negative"
                        ).sum()
                    ),
            ),
            landsat8_count=(
                "satellite_normalized",
                lambda values:
                    int(
                        (
                            values == "LC08"
                        ).sum()
                    ),
            ),
            landsat9_count=(
                "satellite_normalized",
                lambda values:
                    int(
                        (
                            values == "LC09"
                        ).sum()
                    ),
            ),
        )
        .reset_index()
    )

    site_summary = site_summary[
        site_summary["positive_count"].ge(1)
        & site_summary["negative_count"].ge(
            CALIBRATION_NEGATIVES_PER_SITE
            + MODEL_NEGATIVES_PER_SITE
        )
    ].copy()

    # 確認不會碰到已凍結的 33 個外部測試場址。
    external_sites = set(
        external["site_key"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    development_sites = set(
        site_summary["site_key"]
    )

    overlap_with_external = sorted(
        development_sites
        & external_sites
    )

    if overlap_with_external:
        raise RuntimeError(
            "Development/external site overlap:\n"
            + "\n".join(
                overlap_with_external[:20]
            )
        )

    # 依正樣本數建立粗略 strata，
    # 讓 train/validation 的大型與小型場址較平衡。
    site_summary["positive_size_bin"] = pd.cut(
        site_summary["positive_count"],
        bins=[
            0,
            2,
            5,
            10,
            np.inf,
        ],
        labels=[
            "1_to_2",
            "3_to_5",
            "6_to_10",
            "gt_10",
        ],
    ).astype(str)

    strata_counts = (
        site_summary[
            "positive_size_bin"
        ].value_counts()
    )

    rare_strata = set(
        strata_counts[
            strata_counts < 2
        ].index
    )

    site_summary["split_stratum"] = (
        site_summary[
            "positive_size_bin"
        ].where(
            ~site_summary[
                "positive_size_bin"
            ].isin(rare_strata),
            "other",
        )
    )

    # 若合併後仍有只有一個場址的 strata，
    # 就不用 stratify，仍維持固定 random_state。
    final_strata_counts = (
        site_summary[
            "split_stratum"
        ].value_counts()
    )

    stratify_values = (
        site_summary[
            "split_stratum"
        ]
        if (
            len(final_strata_counts) > 1
            and final_strata_counts.min() >= 2
        )
        else None
    )

    train_sites, validation_sites = (
        train_test_split(
            site_summary["site_key"],
            test_size=VALIDATION_FRACTION,
            random_state=RANDOM_STATE,
            stratify=stratify_values,
        )
    )

    train_sites = set(
        train_sites.astype(str)
    )

    validation_sites = set(
        validation_sites.astype(str)
    )

    if train_sites & validation_sites:
        raise RuntimeError(
            "Train/validation site overlap."
        )

    site_summary[
        "development_split"
    ] = np.where(
        site_summary["site_key"].isin(
            validation_sites
        ),
        "validation",
        "train",
    )

    manifest_rows = []

    for site_key in sorted(
        development_sites
    ):
        site_images = images[
            images["site_key"].eq(
                site_key
            )
        ].copy()

        split_name = (
            "validation"
            if site_key in validation_sites
            else "train"
        )

        positives = site_images[
            site_images["benchmark_role"].eq(
                "high_emission_positive"
            )
            & site_images["ch4_fluxrate"].ge(
                HIGH_EMISSION_THRESHOLD_KG_H
            )
        ].copy()

        negatives = site_images[
            site_images["benchmark_role"].eq(
                "no_plume_negative"
            )
        ].copy()

        positives = choose_flux_spread(
            positives,
            MAX_POSITIVES_PER_SITE,
        )

        positive_times = list(
            positives[
                "acquisition_datetime_utc"
            ].dropna()
        )

        positive_sensors = set(
            positives[
                "satellite_normalized"
            ].dropna().astype(str)
        )

        negatives[
            "sensor_matches_positive"
        ] = negatives[
            "satellite_normalized"
        ].astype(str).isin(
            positive_sensors
        )

        negatives[
            "days_to_nearest_positive"
        ] = negatives[
            "acquisition_datetime_utc"
        ].apply(
            lambda timestamp:
                nearest_positive_days(
                    timestamp,
                    positive_times,
                )
        )

        selected_negatives = (
            negatives.sort_values(
                [
                    "sensor_matches_positive",
                    "days_to_nearest_positive",
                    "acquisition_datetime_utc",
                    "image_key",
                ],
                ascending=[
                    False,
                    True,
                    True,
                    True,
                ],
                na_position="last",
            )
            .head(
                CALIBRATION_NEGATIVES_PER_SITE
                + MODEL_NEGATIVES_PER_SITE
            )
            .copy()
        )

        if len(selected_negatives) != 10:
            raise RuntimeError(
                f"{site_key}: expected 10 negatives, "
                f"found {len(selected_negatives)}."
            )

        # 依時間交錯分配，避免 calibration 與 model
        # negatives 集中在不同時期。
        selected_negatives = (
            selected_negatives.sort_values(
                [
                    "acquisition_datetime_utc",
                    "image_key",
                ],
                na_position="last",
            )
            .reset_index(drop=True)
        )

        role_pattern = [
            "calibration_negative",
            "model_negative",
        ] * 5

        selected_negatives[
            "development_role"
        ] = role_pattern

        selected_negatives[
            "evaluation_label"
        ] = 0

        positives[
            "development_role"
        ] = "model_positive"

        positives[
            "evaluation_label"
        ] = 1

        selected = pd.concat(
            [
                selected_negatives,
                positives,
            ],
            ignore_index=True,
            sort=False,
        )

        selected[
            "development_split"
        ] = split_name

        selected[
            "split_version"
        ] = SPLIT_VERSION

        selected[
            "selection_used_model_output"
        ] = False

        manifest_rows.extend(
            selected.to_dict(
                "records"
            )
        )

    manifest = pd.DataFrame(
        manifest_rows
    )

    manifest = manifest.sort_values(
        [
            "development_split",
            "site_key",
            "development_role",
            "acquisition_datetime_utc",
        ],
        na_position="last",
    ).reset_index(drop=True)

    manifest["download_id"] = [
        f"MARS_DEV_{number:05d}"
        for number in range(
            1,
            len(manifest) + 1,
        )
    ]

    # 最終完整性檢查。
    role_table = pd.crosstab(
        manifest["site_key"],
        manifest["development_role"],
    )

    for column in [
        "calibration_negative",
        "model_negative",
        "model_positive",
    ]:
        if column not in role_table.columns:
            role_table[column] = 0

    invalid_sites = role_table[
        ~role_table[
            "calibration_negative"
        ].eq(5)
        | ~role_table[
            "model_negative"
        ].eq(5)
        | role_table[
            "model_positive"
        ].lt(1)
        | role_table[
            "model_positive"
        ].gt(5)
    ]

    if not invalid_sites.empty:
        raise RuntimeError(
            "Invalid site assignments:\n"
            + invalid_sites.to_string()
        )

    summary = (
        manifest.groupby(
            [
                "development_split",
                "development_role",
                "satellite_normalized",
            ],
            dropna=False,
        )
        .agg(
            image_count=(
                "image_key",
                "size",
            ),
            site_count=(
                "site_key",
                "nunique",
            ),
            minimum_flux_kg_h=(
                "ch4_fluxrate",
                "min",
            ),
            median_flux_kg_h=(
                "ch4_fluxrate",
                "median",
            ),
            maximum_flux_kg_h=(
                "ch4_fluxrate",
                "max",
            ),
        )
        .reset_index()
    )

    SITE_SPLIT_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    site_summary.to_csv(
        SITE_SPLIT_OUTPUT,
        index=False,
    )

    manifest.to_csv(
        MANIFEST_OUTPUT,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print("=" * 110)
    print("MARS-S2L SITE-DISJOINT DEVELOPMENT MANIFEST")
    print("=" * 110)

    print(
        "\nUnique eligible sites:",
        len(site_summary),
    )

    print("\nSites by development split:")
    print(
        site_summary[
            "development_split"
        ].value_counts()
    )

    print(
        "\nTrain/validation site overlap:",
        len(
            train_sites
            & validation_sites
        ),
    )

    print(
        "Development/external site overlap:",
        len(
            development_sites
            & external_sites
        ),
    )

    print(
        "\nTotal manifest images:",
        len(manifest),
    )

    print("\nImages by split and role:")
    print(
        pd.crosstab(
            manifest[
                "development_split"
            ],
            manifest[
                "development_role"
            ],
            margins=True,
        )
    )

    print("\nImages by role and sensor:")
    print(
        pd.crosstab(
            manifest[
                "development_role"
            ],
            manifest[
                "satellite_normalized"
            ],
            margins=True,
        )
    )

    print("\nPositive flux summary:")
    print(
        manifest.loc[
            manifest[
                "development_role"
            ].eq("model_positive"),
            "ch4_fluxrate",
        ].describe()
    )

    print("\nSaved:")
    print(SITE_SPLIT_OUTPUT)
    print(MANIFEST_OUTPUT)
    print(SUMMARY_OUTPUT)


if __name__ == "__main__":
    main()
