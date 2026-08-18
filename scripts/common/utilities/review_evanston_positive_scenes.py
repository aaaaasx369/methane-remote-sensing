from pathlib import Path

import pandas as pd


INPUT = Path(
    "outputs/133_evanston_landsat_scene_candidates.csv"
)

OUTPUT = Path(
    "outputs/135_evanston_positive_scene_review.csv"
)

PRIORITY_OUTPUT = Path(
    "outputs/136_evanston_positive_download_manifest.csv"
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


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    if df.empty:
        raise RuntimeError(
            "Scene candidate table is empty."
        )

    df["exact_release_overlap"] = (
        parse_boolean(
            df["exact_release_overlap"]
        )
    )

    df["cloud_cover"] = pd.to_numeric(
        df["cloud_cover"],
        errors="coerce",
    )

    df["flow_max_kg_h"] = pd.to_numeric(
        df["flow_max_kg_h"],
        errors="coerce",
    )

    df["minutes_from_release_window"] = (
        pd.to_numeric(
            df["minutes_from_release_window"],
            errors="coerce",
        )
    )

    df["has_product_id"] = (
        df["landsat_product_id"]
        .notna()
        & df["landsat_product_id"]
        .astype(str)
        .str.strip()
        .ne("")
    )

    df["sensor_matches"] = (
        (
            df["expected_sensor"]
            .eq("Landsat-8")
            & df["spacecraft_id"]
            .astype(str)
            .str.contains(
                "LANDSAT_8",
                case=False,
                na=False,
            )
        )
        |
        (
            df["expected_sensor"]
            .eq("Landsat-9")
            & df["spacecraft_id"]
            .astype(str)
            .str.contains(
                "LANDSAT_9",
                case=False,
                na=False,
            )
        )
    )

    decisions = []

    for _, row in df.iterrows():
        if not row["has_product_id"]:
            decision = (
                "exclude_missing_product"
            )

        elif row["exact_release_overlap"] is not True:
            decision = (
                "exclude_no_overlap"
            )

        elif row["sensor_matches"] is not True:
            decision = (
                "manual_review_sensor"
            )

        elif (
            pd.notna(row["cloud_cover"])
            and row["cloud_cover"] <= 30
        ):
            decision = (
                "priority_download"
            )

        else:
            decision = (
                "manual_review_cloud"
            )

        decisions.append(decision)

    df["review_decision"] = decisions

    # 同一個產品只能保留一次，避免重複下載。
    df["duplicate_product"] = df.duplicated(
        subset=["landsat_product_id"],
        keep="first",
    )

    df.loc[
        df["duplicate_product"]
        & df["review_decision"].isin([
            "priority_download",
            "manual_review_cloud",
        ]),
        "review_decision",
    ] = "exclude_duplicate_product"

    df = df.sort_values(
        [
            "review_decision",
            "cloud_cover",
            "flow_max_kg_h",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    ).reset_index(drop=True)

    df.to_csv(
        OUTPUT,
        index=False,
    )

    priority = df[
        df["review_decision"]
        == "priority_download"
    ].copy()

    priority["label"] = 1
    priority["site_key"] = "evanston"
    priority["ground_truth_type"] = (
        "controlled_release_exact_overlap"
    )

    priority.to_csv(
        PRIORITY_OUTPUT,
        index=False,
    )

    print("=" * 105)
    print("EVANSTON POSITIVE SCENE REVIEW")
    print("=" * 105)

    print("\nDecision counts:")
    print(
        df["review_decision"]
        .value_counts(dropna=False)
    )

    print("\nPriority downloads:", len(priority))

    if len(priority):
        print(
            priority[
                [
                    "overpass_id",
                    "landsat_product_id",
                    "expected_sensor",
                    "acquisition_time_utc",
                    "cloud_cover",
                    "flow_max_kg_h",
                    "exact_release_overlap",
                ]
            ].to_string(index=False)
        )

    print("\nSaved:")
    print(OUTPUT)
    print(PRIORITY_OUTPUT)


if __name__ == "__main__":
    main()
