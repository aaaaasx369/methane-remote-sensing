from pathlib import Path

import numpy as np
import pandas as pd


GROUND_TRUTH_ROOT = Path(
    "raw_data/stanford_large_scale_release/"
    "evanston_landsat_ground_truth"
)

SCENE_INPUT = Path(
    "outputs/133_evanston_landsat_scene_candidates.csv"
)

WINDOW_OUTPUT = Path(
    "outputs/138_evanston_detailed_release_windows.csv"
)

SCENE_REVIEW_OUTPUT = Path(
    "outputs/139_evanston_scene_overlap_revised.csv"
)

POSITIVE_OUTPUT = Path(
    "outputs/140_evanston_confirmed_positive_download_manifest.csv"
)

# 高於 1 kg/h 才視為 valve-on，避免極小數值雜訊。
FLOW_ON_THRESHOLD_KG_H = 1.0

# 允許釋放過程中最多 5 秒的短暫資料缺口或瞬間零值。
MAX_POSITIVE_GAP_SECONDS = 5

# 這是整景 CLOUD_COVER，只用於下載優先級；
# 真正是否有雲仍需下載後檢查排放源附近 QA pixel。
PRIORITY_CLOUD_LIMIT = 60.0


def parse_datetime(
    date_series,
    time_series,
):
    return pd.to_datetime(
        date_series.astype(str).str.strip()
        + " "
        + time_series.astype(str).str.strip(),
        errors="coerce",
        utc=True,
    )


def read_summary(folder, release_id):
    expected = folder / f"{release_id}_summary.csv"

    if expected.exists():
        path = expected
    else:
        matches = sorted(
            folder.glob("*_summary.csv")
        )

        if not matches:
            return None, None

        path = matches[0]

    summary = pd.read_csv(
        path,
        low_memory=False,
    )

    if summary.empty:
        return None, path

    row = summary.iloc[0].copy()

    summary_time = parse_datetime(
        pd.Series([row.get("date")]),
        pd.Series([row.get("time_UTC")]),
    ).iloc[0]

    row["summary_datetime_utc"] = (
        summary_time
    )

    return row, path


def build_positive_intervals(
    releasedata,
):
    positive = releasedata[
        releasedata["flow_kg_h"]
        > FLOW_ON_THRESHOLD_KG_H
    ].copy()

    if positive.empty:
        return pd.DataFrame()

    time_gap = (
        positive["datetime_utc"]
        .diff()
        .dt.total_seconds()
    )

    new_group = (
        time_gap.isna()
        | (
            time_gap
            > MAX_POSITIVE_GAP_SECONDS
        )
    )

    positive["interval_group"] = (
        new_group.cumsum()
    )

    intervals = (
        positive.groupby(
            "interval_group"
        )
        .agg(
            release_start_utc=(
                "datetime_utc",
                "min",
            ),
            release_end_utc=(
                "datetime_utc",
                "max",
            ),
            positive_samples=(
                "flow_kg_h",
                "size",
            ),
            flow_mean_kg_h=(
                "flow_kg_h",
                "mean",
            ),
            flow_median_kg_h=(
                "flow_kg_h",
                "median",
            ),
            flow_min_kg_h=(
                "flow_kg_h",
                "min",
            ),
            flow_max_kg_h=(
                "flow_kg_h",
                "max",
            ),
        )
        .reset_index()
    )

    intervals["duration_seconds"] = (
        intervals["release_end_utc"]
        - intervals["release_start_utc"]
    ).dt.total_seconds() + 1

    intervals["interval_midpoint_utc"] = (
        intervals["release_start_utc"]
        + (
            intervals["release_end_utc"]
            - intervals["release_start_utc"]
        ) / 2
    )

    return intervals


def choose_summary_interval(
    intervals,
    summary_time,
):
    if intervals.empty:
        return None

    intervals = intervals.copy()

    if pd.notna(summary_time):
        intervals[
            "contains_summary_time"
        ] = (
            intervals[
                "release_start_utc"
            ].le(summary_time)
            & intervals[
                "release_end_utc"
            ].ge(summary_time)
        )

        intervals[
            "seconds_from_summary"
        ] = (
            intervals[
                "interval_midpoint_utc"
            ]
            - summary_time
        ).abs().dt.total_seconds()

        containing = intervals[
            intervals[
                "contains_summary_time"
            ]
        ]

        if not containing.empty:
            return (
                containing.sort_values(
                    "duration_seconds",
                    ascending=False,
                ).iloc[0]
            )

        return (
            intervals.sort_values(
                "seconds_from_summary"
            ).iloc[0]
        )

    return (
        intervals.sort_values(
            "duration_seconds",
            ascending=False,
        ).iloc[0]
    )


def nearest_flow_statistics(
    releasedata,
    acquisition_time,
):
    if (
        releasedata.empty
        or pd.isna(acquisition_time)
    ):
        return {
            "flow_at_scene_kg_h": np.nan,
            "flow_nearest_time_utc": pd.NaT,
            "flow_nearest_distance_seconds":
                np.nan,
            "flow_mean_pm30s_kg_h": np.nan,
            "flow_mean_pm300s_kg_h": np.nan,
        }

    time_difference = (
        releasedata["datetime_utc"]
        - acquisition_time
    ).abs()

    nearest_index = (
        time_difference.idxmin()
    )

    nearest_row = releasedata.loc[
        nearest_index
    ]

    distance_seconds = float(
        time_difference.loc[
            nearest_index
        ].total_seconds()
    )

    window_30 = releasedata[
        releasedata["datetime_utc"].between(
            acquisition_time
            - pd.Timedelta(seconds=30),
            acquisition_time
            + pd.Timedelta(seconds=30),
        )
    ]

    window_300 = releasedata[
        releasedata["datetime_utc"].between(
            acquisition_time
            - pd.Timedelta(seconds=300),
            acquisition_time
            + pd.Timedelta(seconds=300),
        )
    ]

    return {
        "flow_at_scene_kg_h":
            float(
                nearest_row["flow_kg_h"]
            ),
        "flow_nearest_time_utc":
            nearest_row[
                "datetime_utc"
            ],
        "flow_nearest_distance_seconds":
            distance_seconds,
        "flow_mean_pm30s_kg_h":
            (
                float(
                    window_30[
                        "flow_kg_h"
                    ].mean()
                )
                if not window_30.empty
                else np.nan
            ),
        "flow_mean_pm300s_kg_h":
            (
                float(
                    window_300[
                        "flow_kg_h"
                    ].mean()
                )
                if not window_300.empty
                else np.nan
            ),
    }


def find_matching_release_id(
    scene_row,
    known_release_ids,
):
    possible_ids = []

    representative = scene_row.get(
        "representative_release_id"
    )

    if pd.notna(representative):
        possible_ids.append(
            str(representative).strip()
        )

    all_ids = scene_row.get(
        "all_release_ids"
    )

    if pd.notna(all_ids):
        possible_ids.extend(
            value.strip()
            for value
            in str(all_ids).split("|")
            if value.strip()
        )

    for release_id in possible_ids:
        if release_id in known_release_ids:
            return release_id

    return ""


def main():
    release_files = sorted(
        GROUND_TRUTH_ROOT.rglob(
            "*_releasedata.csv"
        )
    )

    print("=" * 105)
    print("REBUILDING EVANSTON RELEASE WINDOWS")
    print("=" * 105)

    print(
        "Detailed release files:",
        len(release_files),
    )

    window_rows = []
    release_data_by_id = {}

    for path in release_files:
        release_id = path.stem.replace(
            "_releasedata",
            "",
        )

        try:
            releasedata = pd.read_csv(
                path,
                low_memory=False,
            )

            required = [
                "date",
                "time_UTC",
                "methaneflow_KGperHR",
            ]

            missing = [
                column
                for column in required
                if column
                not in releasedata.columns
            ]

            if missing:
                raise KeyError(
                    f"Missing columns: {missing}"
                )

            releasedata["datetime_utc"] = (
                parse_datetime(
                    releasedata["date"],
                    releasedata["time_UTC"],
                )
            )

            releasedata["flow_kg_h"] = (
                pd.to_numeric(
                    releasedata[
                        "methaneflow_KGperHR"
                    ],
                    errors="coerce",
                )
            )

            releasedata = (
                releasedata.dropna(
                    subset=[
                        "datetime_utc",
                        "flow_kg_h",
                    ]
                )
                .sort_values(
                    "datetime_utc"
                )
                .reset_index(drop=True)
            )

            summary, summary_path = (
                read_summary(
                    path.parent,
                    release_id,
                )
            )

            summary_time = (
                summary.get(
                    "summary_datetime_utc"
                )
                if summary is not None
                else pd.NaT
            )

            intervals = (
                build_positive_intervals(
                    releasedata
                )
            )

            selected = (
                choose_summary_interval(
                    intervals,
                    summary_time,
                )
            )

            if selected is None:
                print(
                    f"[NO POSITIVE FLOW] "
                    f"{release_id}"
                )

                window_rows.append({
                    "release_id":
                        release_id,
                    "status":
                        "no_positive_flow",
                    "releasedata_path":
                        str(path),
                    "summary_path":
                        (
                            str(summary_path)
                            if summary_path
                            else ""
                        ),
                })

                continue

            release_data_by_id[
                release_id
            ] = releasedata

            row = {
                "release_id":
                    release_id,
                "status":
                    "success",
                "releasedata_path":
                    str(path),
                "summary_path":
                    (
                        str(summary_path)
                        if summary_path
                        else ""
                    ),
                "summary_datetime_utc":
                    summary_time,
                "release_start_utc":
                    selected[
                        "release_start_utc"
                    ],
                "release_end_utc":
                    selected[
                        "release_end_utc"
                    ],
                "duration_seconds":
                    selected[
                        "duration_seconds"
                    ],
                "positive_samples":
                    selected[
                        "positive_samples"
                    ],
                "flow_mean_kg_h":
                    selected[
                        "flow_mean_kg_h"
                    ],
                "flow_median_kg_h":
                    selected[
                        "flow_median_kg_h"
                    ],
                "flow_min_kg_h":
                    selected[
                        "flow_min_kg_h"
                    ],
                "flow_max_kg_h":
                    selected[
                        "flow_max_kg_h"
                    ],
                "summary_inside_interval":
                    (
                        pd.notna(summary_time)
                        and selected[
                            "release_start_utc"
                        ]
                        <= summary_time
                        <= selected[
                            "release_end_utc"
                        ]
                    ),
                "interval_count":
                    len(intervals),
                "total_data_rows":
                    len(releasedata),
            }

            if summary is not None:
                for column in [
                    "location",
                    "lat",
                    "lon",
                    "ch4_kgh_mean",
                    "ch4_kgh_sigma",
                ]:
                    row[
                        f"summary_{column}"
                    ] = summary.get(column)

            window_rows.append(row)

            print(
                f"[OK] {release_id} | "
                f"{selected['release_start_utc']} "
                f"to "
                f"{selected['release_end_utc']} | "
                f"mean="
                f"{selected['flow_mean_kg_h']:.1f} "
                f"kg/h"
            )

        except Exception as error:
            print(
                f"[ERROR] {release_id} | "
                f"{error}"
            )

            window_rows.append({
                "release_id":
                    release_id,
                "status":
                    "error",
                "error":
                    str(error),
                "releasedata_path":
                    str(path),
            })

    windows = pd.DataFrame(
        window_rows
    )

    windows.to_csv(
        WINDOW_OUTPUT,
        index=False,
    )

    if not SCENE_INPUT.exists():
        raise FileNotFoundError(
            SCENE_INPUT
        )

    scenes = pd.read_csv(
        SCENE_INPUT,
        low_memory=False,
    )

    scenes["acquisition_time_parsed"] = (
        pd.to_datetime(
            scenes[
                "acquisition_time_utc"
            ],
            errors="coerce",
            utc=True,
        )
    )

    successful_windows = windows[
        windows["status"] == "success"
    ].copy()

    successful_windows[
        "release_start_parsed"
    ] = pd.to_datetime(
        successful_windows[
            "release_start_utc"
        ],
        errors="coerce",
        utc=True,
    )

    successful_windows[
        "release_end_parsed"
    ] = pd.to_datetime(
        successful_windows[
            "release_end_utc"
        ],
        errors="coerce",
        utc=True,
    )

    window_lookup = (
        successful_windows
        .set_index("release_id")
        .to_dict("index")
    )

    known_release_ids = set(
        window_lookup
    )

    reviewed_rows = []

    for _, scene in scenes.iterrows():
        row = scene.to_dict()

        release_id = (
            find_matching_release_id(
                scene,
                known_release_ids,
            )
        )

        row[
            "matched_release_id"
        ] = release_id

        if not release_id:
            row[
                "detailed_release_status"
            ] = "no_matching_release_file"

            reviewed_rows.append(row)
            continue

        release_window = (
            window_lookup[
                release_id
            ]
        )

        acquisition_time = scene[
            "acquisition_time_parsed"
        ]

        release_start = (
            release_window[
                "release_start_parsed"
            ]
        )

        release_end = (
            release_window[
                "release_end_parsed"
            ]
        )

        exact_overlap = bool(
            pd.notna(acquisition_time)
            and release_start
            <= acquisition_time
            <= release_end
        )

        flow_stats = (
            nearest_flow_statistics(
                release_data_by_id[
                    release_id
                ],
                acquisition_time,
            )
        )

        row.update({
            "detailed_release_status":
                "matched",
            "detailed_release_start_utc":
                release_start,
            "detailed_release_end_utc":
                release_end,
            "detailed_release_duration_seconds":
                release_window[
                    "duration_seconds"
                ],
            "detailed_flow_mean_kg_h":
                release_window[
                    "flow_mean_kg_h"
                ],
            "detailed_flow_max_kg_h":
                release_window[
                    "flow_max_kg_h"
                ],
            "exact_detailed_flow_overlap":
                exact_overlap,
            **flow_stats,
        })

        reviewed_rows.append(row)

    review = pd.DataFrame(
        reviewed_rows
    )

    review["cloud_cover"] = (
        pd.to_numeric(
            review["cloud_cover"],
            errors="coerce",
        )
    )

    # 同一 release、同一衛星時間可能有相鄰 WRS scenes。
    # 優先保留 CLOUD_COVER 較低者。
    review = review.sort_values(
        [
            "matched_release_id",
            "acquisition_time_utc",
            "cloud_cover",
        ],
        na_position="last",
    ).reset_index(drop=True)

    review[
        "same_release_time_rank"
    ] = (
        review.groupby(
            [
                "matched_release_id",
                "acquisition_time_utc",
            ],
            dropna=False,
        )
        .cumcount()
        + 1
    )

    decisions = []

    for _, row in review.iterrows():
        if (
            row.get(
                "detailed_release_status"
            )
            != "matched"
        ):
            decision = (
                "exclude_no_detailed_ground_truth"
            )

        elif (
            row.get(
                "exact_detailed_flow_overlap"
            )
            is not True
        ):
            decision = (
                "exclude_no_flow_overlap"
            )

        elif (
            row.get(
                "same_release_time_rank"
            )
            > 1
        ):
            decision = (
                "exclude_duplicate_overlapping_scene"
            )

        elif (
            pd.isna(row.get("cloud_cover"))
            or row.get("cloud_cover")
            > PRIORITY_CLOUD_LIMIT
        ):
            decision = (
                "manual_review_cloud"
            )

        else:
            decision = (
                "priority_download"
            )

        decisions.append(decision)

    review[
        "revised_review_decision"
    ] = decisions

    review.to_csv(
        SCENE_REVIEW_OUTPUT,
        index=False,
    )

    positives = review[
        review[
            "revised_review_decision"
        ] == "priority_download"
    ].copy()

    positives["label"] = 1
    positives["site_key"] = "evanston"
    positives["ground_truth_type"] = (
        "controlled_release_"
        "detailed_flow_exact_overlap"
    )

    positives.to_csv(
        POSITIVE_OUTPUT,
        index=False,
    )

    print("\n" + "=" * 105)
    print("REVISED SCENE OVERLAP SUMMARY")
    print("=" * 105)

    print("\nDecision counts:")
    print(
        review[
            "revised_review_decision"
        ].value_counts(
            dropna=False
        )
    )

    display_columns = [
        "overpass_id",
        "matched_release_id",
        "landsat_product_id",
        "acquisition_time_utc",
        "detailed_release_start_utc",
        "detailed_release_end_utc",
        "exact_detailed_flow_overlap",
        "flow_at_scene_kg_h",
        "flow_mean_pm30s_kg_h",
        "cloud_cover",
        "revised_review_decision",
    ]

    print("\nRevised scene review:")
    print(
        review[
            [
                column
                for column in display_columns
                if column in review.columns
            ]
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.2f}",
        )
    )

    print("\nPriority positive downloads:")
    print(len(positives))

    print("\nSaved:")
    print(WINDOW_OUTPUT)
    print(SCENE_REVIEW_OUTPUT)
    print(POSITIVE_OUTPUT)


if __name__ == "__main__":
    main()
