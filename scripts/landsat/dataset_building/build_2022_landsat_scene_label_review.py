from pathlib import Path

import numpy as np
import pandas as pd


CANDIDATE_CSV = Path(
    "outputs/51_2022_landsat_release_interval_candidates.csv"
)

SCENE_CSV = Path(
    "outputs/43_landsat_unique_clean_features.csv"
)

OUTPUT_CSV = Path(
    "outputs/52_2022_landsat_scene_label_review.csv"
)


def parse_boolean(series):
    """
    Robustly convert True/False strings or booleans.
    """
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
    if not CANDIDATE_CSV.exists():
        raise FileNotFoundError(
            f"Missing candidate file: {CANDIDATE_CSV}"
        )

    if not SCENE_CSV.exists():
        raise FileNotFoundError(
            f"Missing scene file: {SCENE_CSV}"
        )

    candidates = pd.read_csv(CANDIDATE_CSV)
    scenes = pd.read_csv(SCENE_CSV)

    candidates["overlaps_landsat"] = parse_boolean(
        candidates["overlaps_landsat"]
    )

    candidates["ch4_kgh_mean"] = pd.to_numeric(
        candidates["ch4_kgh_mean"],
        errors="coerce",
    )

    candidates["seconds_to_interval"] = pd.to_numeric(
        candidates["seconds_to_interval"],
        errors="coerce",
    )

    scenes["label"] = pd.to_numeric(
        scenes["label"],
        errors="coerce",
    )

    scenes_2022 = scenes[
        pd.to_datetime(
            scenes["landsat_image_time"],
            errors="coerce",
        ).dt.year == 2022
    ].copy()

    review_rows = []

    for _, scene in scenes_2022.iterrows():
        raster_group_id = scene["raster_group_id"]

        scene_candidates = candidates[
            candidates["raster_group_id"]
            == raster_group_id
        ].copy()

        overlapping = scene_candidates[
            scene_candidates["overlaps_landsat"] == True
        ].copy()

        # Prefer an overlapping measurement window.
        if len(overlapping) > 0:
            selected = overlapping.sort_values(
                by=[
                    "seconds_to_interval",
                    "candidate_rank",
                ]
            ).iloc[0]

            rate = selected["ch4_kgh_mean"]

            if pd.isna(rate):
                review_status = "ambiguous_missing_release_rate"
                recommended_label = np.nan
                reason = (
                    "The Landsat acquisition overlaps a release-rate "
                    "window, but ch4_kgh_mean is missing."
                )

            elif rate > 0:
                review_status = "confirmed_positive"
                recommended_label = 1
                reason = (
                    "The Landsat acquisition overlaps the measurement "
                    f"window and ch4_kgh_mean={rate:.6f} kg/h."
                )

            else:
                review_status = "confirmed_negative"
                recommended_label = 0
                reason = (
                    "The Landsat acquisition overlaps the measurement "
                    "window and ch4_kgh_mean is 0 kg/h."
                )

        else:
            # No time-overlapping measurement exists.
            if len(scene_candidates) > 0:
                selected = scene_candidates.sort_values(
                    by=[
                        "seconds_to_interval",
                        "candidate_rank",
                    ]
                ).iloc[0]

                gap_hours = (
                    selected["seconds_to_interval"]
                    / 3600
                )

                reason = (
                    "No release-rate window overlaps the Landsat "
                    f"acquisition. Nearest interval is "
                    f"{gap_hours:.6f} hours away."
                )

            else:
                selected = pd.Series(dtype=object)

                reason = (
                    "No release-rate candidate was found for this scene."
                )

            review_status = "ambiguous_no_overlapping_measurement"
            recommended_label = np.nan

        current_label = scene["label"]

        if pd.isna(recommended_label):
            label_action = "exclude_pending_review"

        elif int(current_label) == int(recommended_label):
            label_action = "keep_current_label"

        else:
            label_action = (
                f"change_{int(current_label)}"
                f"_to_{int(recommended_label)}"
            )

        review_rows.append({
            "raster_group_id": raster_group_id,
            "landsat_image_time":
                scene.get("landsat_image_time"),
            "landsat_sensor":
                scene.get("landsat_sensor"),
            "current_label": current_label,
            "recommended_label": recommended_label,
            "review_status": review_status,
            "label_action": label_action,
            "release_start_utc":
                selected.get("release_start_utc"),
            "release_end_utc":
                selected.get("release_end_utc"),
            "overlaps_landsat":
                selected.get("overlaps_landsat"),
            "seconds_to_interval":
                selected.get("seconds_to_interval"),
            "ch4_kgh_mean":
                selected.get("ch4_kgh_mean"),
            "ch4_kgh_sigma":
                selected.get("ch4_kgh_sigma"),
            "review_reason": reason,
        })

    review_df = pd.DataFrame(review_rows)

    review_df = review_df.sort_values(
        by=[
            "review_status",
            "raster_group_id",
        ]
    ).reset_index(drop=True)

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    review_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print("=" * 90)
    print("2022 LANDSAT SCENE-LEVEL LABEL REVIEW")
    print("=" * 90)

    print(f"\nScenes reviewed: {len(review_df)}")

    print("\nReview-status counts:")
    print(
        review_df["review_status"]
        .value_counts()
    )

    print("\nRecommended confirmed-label counts:")
    print(
        review_df["recommended_label"]
        .value_counts(dropna=False)
        .sort_index()
    )

    print("\nLabel actions:")
    print(
        review_df["label_action"]
        .value_counts()
    )

    print("\nComplete review:")
    print(
        review_df[
            [
                "raster_group_id",
                "landsat_image_time",
                "current_label",
                "recommended_label",
                "review_status",
                "label_action",
                "ch4_kgh_mean",
                "seconds_to_interval",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
