from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "outputs/206_carbonmapper_ch4_ge1000_timed_candidates.csv"
)

SCENE_OUTPUT = Path(
    "outputs/209_carbonmapper_tanager_scene_candidates.csv"
)

PLUME_LINK_OUTPUT = Path(
    "outputs/210_carbonmapper_tanager_scene_plume_links.csv"
)

SUMMARY_OUTPUT = Path(
    "outputs/211_carbonmapper_tanager_scene_summary.csv"
)


def valid_text(series):
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .replace(
            {
                "nan": "",
                "None": "",
            }
        )
    )


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    required = [
        "plume_id",
        "scene_id",
        "scene_datetime_utc",
        "instrument",
        "plume_latitude",
        "plume_longitude",
        "emission_auto",
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

    df["scene_datetime_utc"] = pd.to_datetime(
        df["scene_datetime_utc"],
        errors="coerce",
        utc=True,
    )

    df["plume_latitude"] = pd.to_numeric(
        df["plume_latitude"],
        errors="coerce",
    )

    df["plume_longitude"] = pd.to_numeric(
        df["plume_longitude"],
        errors="coerce",
    )

    df["emission_auto"] = pd.to_numeric(
        df["emission_auto"],
        errors="coerce",
    )

    if "emission_uncertainty_auto" in df.columns:
        df["emission_uncertainty_auto"] = (
            pd.to_numeric(
                df["emission_uncertainty_auto"],
                errors="coerce",
            )
        )
    else:
        df["emission_uncertainty_auto"] = np.nan

    df["relative_emission_uncertainty"] = (
        df["emission_uncertainty_auto"]
        / df["emission_auto"]
    )

    df["scene_id_clean"] = valid_text(
        df["scene_id"]
    )

    # scene_id 缺失時，以 instrument + 秒級時間作為替代鍵值。
    fallback_scene_key = (
        df["instrument"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        + "|"
        + df["scene_datetime_utc"]
        .dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        .fillna("missing_time")
    )

    df["scene_key"] = np.where(
        df["scene_id_clean"].ne(""),
        df["scene_id_clean"],
        fallback_scene_key,
    )

    df = df.dropna(
        subset=[
            "scene_datetime_utc",
            "plume_latitude",
            "plume_longitude",
            "emission_auto",
        ]
    ).copy()

    # 每個 scene 的最高排放 plume 作為代表 plume。
    df = df.sort_values(
        [
            "scene_key",
            "emission_auto",
            "relative_emission_uncertainty",
        ],
        ascending=[
            True,
            False,
            True,
        ],
        na_position="last",
    )

    df["plume_rank_within_scene"] = (
        df.groupby("scene_key")
        .cumcount()
        + 1
    )

    representatives = df[
        df["plume_rank_within_scene"] == 1
    ][
        [
            "scene_key",
            "plume_id",
            "plume_latitude",
            "plume_longitude",
            "emission_auto",
            "emission_uncertainty_auto",
            "relative_emission_uncertainty",
            "plume_tif",
            "plume_png",
            "plume_rgb_png",
            "rgb_png",
        ]
    ].copy()

    representatives = representatives.rename(
        columns={
            "plume_id":
                "representative_plume_id",
            "plume_latitude":
                "representative_latitude",
            "plume_longitude":
                "representative_longitude",
            "emission_auto":
                "representative_emission_kg_h",
            "emission_uncertainty_auto":
                "representative_uncertainty_kg_h",
            "relative_emission_uncertainty":
                "representative_relative_uncertainty",
            "plume_tif":
                "representative_plume_tif",
            "plume_png":
                "representative_plume_png",
            "plume_rgb_png":
                "representative_plume_rgb_png",
            "rgb_png":
                "representative_rgb_png",
        }
    )

    scene_summary = (
        df.groupby(
            "scene_key",
            dropna=False,
        )
        .agg(
            scene_id=(
                "scene_id_clean",
                "first",
            ),
            scene_datetime_utc=(
                "scene_datetime_utc",
                "first",
            ),
            instrument=(
                "instrument",
                "first",
            ),
            platform=(
                "platform",
                "first",
            ),
            plume_count=(
                "plume_id",
                "nunique",
            ),
            scene_median_latitude=(
                "plume_latitude",
                "median",
            ),
            scene_median_longitude=(
                "plume_longitude",
                "median",
            ),
            minimum_emission_kg_h=(
                "emission_auto",
                "min",
            ),
            median_emission_kg_h=(
                "emission_auto",
                "median",
            ),
            maximum_emission_kg_h=(
                "emission_auto",
                "max",
            ),
            uncertainty_available_count=(
                "emission_uncertainty_auto",
                "count",
            ),
        )
        .reset_index()
    )

    scene_summary = scene_summary.merge(
        representatives,
        on="scene_key",
        how="left",
        validate="one_to_one",
    )

    scene_summary["landsat_search_status"] = (
        "not_searched"
    )

    scene_summary["candidate_ground_truth_status"] = (
        "carbon_mapper_reported_plume_"
        "manual_review_pending"
    )

    scene_summary = scene_summary.sort_values(
        [
            "maximum_emission_kg_h",
            "scene_datetime_utc",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)

    df.to_csv(
        PLUME_LINK_OUTPUT,
        index=False,
    )

    scene_summary.to_csv(
        SCENE_OUTPUT,
        index=False,
    )

    summary = pd.DataFrame([
        {
            "eligible_plume_rows":
                len(df),
            "unique_tanager_scenes":
                scene_summary[
                    "scene_key"
                ].nunique(),
            "scenes_with_multiple_plumes":
                int(
                    (
                        scene_summary[
                            "plume_count"
                        ] > 1
                    ).sum()
                ),
            "maximum_plumes_in_one_scene":
                int(
                    scene_summary[
                        "plume_count"
                    ].max()
                ),
            "scenes_with_uncertainty":
                int(
                    (
                        scene_summary[
                            "uncertainty_available_count"
                        ] > 0
                    ).sum()
                ),
            "first_scene":
                scene_summary[
                    "scene_datetime_utc"
                ].min(),
            "last_scene":
                scene_summary[
                    "scene_datetime_utc"
                ].max(),
            "minimum_scene_max_emission_kg_h":
                scene_summary[
                    "maximum_emission_kg_h"
                ].min(),
            "median_scene_max_emission_kg_h":
                scene_summary[
                    "maximum_emission_kg_h"
                ].median(),
            "maximum_scene_emission_kg_h":
                scene_summary[
                    "maximum_emission_kg_h"
                ].max(),
        }
    ])

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print("=" * 105)
    print("CARBON MAPPER TANAGER SCENE PREPARATION")
    print("=" * 105)

    print(
        "\nEligible plume rows:",
        len(df),
    )

    print(
        "Unique Tanager scenes:",
        scene_summary[
            "scene_key"
        ].nunique(),
    )

    print(
        "Scenes with multiple plumes:",
        int(
            (
                scene_summary[
                    "plume_count"
                ] > 1
            ).sum()
        ),
    )

    print(
        "Maximum plumes in one scene:",
        int(
            scene_summary[
                "plume_count"
            ].max()
        ),
    )

    print(
        "Scenes with emission uncertainty:",
        int(
            (
                scene_summary[
                    "uncertainty_available_count"
                ] > 0
            ).sum()
        ),
    )

    print("\nScene date range:")
    print(
        scene_summary[
            "scene_datetime_utc"
        ].min()
    )
    print(
        scene_summary[
            "scene_datetime_utc"
        ].max()
    )

    print("\nPlume count per scene:")
    print(
        scene_summary[
            "plume_count"
        ].describe()
    )

    print("\nMaximum emission per scene:")
    print(
        scene_summary[
            "maximum_emission_kg_h"
        ].describe()
    )

    print("\nTop 15 scenes:")
    display_columns = [
        "scene_key",
        "scene_datetime_utc",
        "plume_count",
        "representative_plume_id",
        "representative_emission_kg_h",
        "representative_relative_uncertainty",
        "scene_median_latitude",
        "scene_median_longitude",
    ]

    print(
        scene_summary[
            display_columns
        ].head(15).to_string(
            index=False,
            float_format=lambda value:
                f"{value:.3f}",
        )
    )

    print("\nSaved:")
    print(SCENE_OUTPUT)
    print(PLUME_LINK_OUTPUT)
    print(SUMMARY_OUTPUT)


if __name__ == "__main__":
    main()
