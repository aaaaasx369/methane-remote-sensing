from pathlib import Path
import re

import numpy as np
import pandas as pd


CANDIDATE_INPUT = Path(
    "outputs/405_casa_grande_new_independent_candidates_v1.csv"
)

RELEASE_INTERVAL_INPUT = Path(
    "outputs/309_all_exact_release_intervals_for_s2.csv"
)

POSITIVE_INPUT = Path(
    "outputs/396_landsat_final_confirmed_features_site_repaired_v1.csv"
)

AUDIT_OUTPUT = Path(
    "outputs/406_casa_grande_expanded_release_audit_v1.csv"
)

CLEAN_EXACT_OUTPUT = Path(
    "outputs/407_casa_grande_expanded_clean_exact_v1.csv"
)

CLEAN_24H_OUTPUT = Path(
    "outputs/408_casa_grande_expanded_clean_24h_v1.csv"
)

REPORT_OUTPUT = Path(
    "outputs/409_casa_grande_expanded_release_report_v1.txt"
)


SAFETY_BUFFER_HOURS = 24.0
NEGATIVES_PER_POSITIVE = 4


def find_column(
    frame,
    candidates,
    table_name,
    required=True,
):
    for column in candidates:
        if column in frame.columns:
            return column

    if required:
        raise KeyError(
            f"{table_name} 找不到欄位："
            + ", ".join(candidates)
        )

    return None


def normalize_site_alias(value):
    text = str(value).strip().lower()

    if "casa" in text:
        return "casa_grande"

    if "ehrenberg" in text:
        return "ehrenberg"

    if text in {
        "",
        "nan",
        "none",
        "<na>",
    }:
        return pd.NA

    return re.sub(
        r"[^a-z0-9]+",
        "_",
        text,
    ).strip("_")


def parse_bool(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin([
            "true",
            "1",
            "yes",
        ])
    )


def interval_distance_hours(
    acquisition_time,
    start_time,
    end_time,
):
    if (
        pd.isna(acquisition_time)
        or pd.isna(start_time)
        or pd.isna(end_time)
    ):
        return np.nan

    if (
        start_time
        <= acquisition_time
        <= end_time
    ):
        return 0.0

    if acquisition_time < start_time:
        return (
            start_time
            - acquisition_time
        ).total_seconds() / 3600.0

    return (
        acquisition_time
        - end_time
    ).total_seconds() / 3600.0


def load_release_intervals():
    if not RELEASE_INTERVAL_INPUT.exists():
        raise FileNotFoundError(
            RELEASE_INTERVAL_INPUT
        )

    frame = pd.read_csv(
        RELEASE_INTERVAL_INPUT,
        low_memory=False,
    )

    site_column = find_column(
        frame,
        [
            "site",
            "site_name",
            "site_key",
            "release_site",
            "site_name_normalized",
        ],
        "Release interval table",
    )

    start_column = find_column(
        frame,
        [
            "release_start_utc",
            "exact_release_start_utc",
            "interval_start_utc",
            "start_time_utc",
            "start_utc",
        ],
        "Release interval table",
    )

    end_column = find_column(
        frame,
        [
            "release_end_utc",
            "exact_release_end_utc",
            "interval_end_utc",
            "end_time_utc",
            "end_utc",
        ],
        "Release interval table",
    )

    rate_column = find_column(
        frame,
        [
            "release_rate_kg_h",
            "preferred_release_rate_kg_h",
            "final_release_rate_kg_h",
            "exact_release_rate_kg_h",
            "selected_release_rate_kg_h",
            "cr_kgh_CH4_mean300",
            "ch4_kgh_mean",
        ],
        "Release interval table",
    )

    interval_id_column = find_column(
        frame,
        [
            "release_interval_id",
            "interval_id",
            "event_id",
            "selected_release_interval_id",
        ],
        "Release interval table",
        required=False,
    )

    result = pd.DataFrame({
        "site_alias":
            frame[
                site_column
            ].map(
                normalize_site_alias
            ),

        "release_start_utc":
            pd.to_datetime(
                frame[start_column],
                errors="coerce",
                utc=True,
            ),

        "release_end_utc":
            pd.to_datetime(
                frame[end_column],
                errors="coerce",
                utc=True,
            ),

        "release_rate_kg_h":
            pd.to_numeric(
                frame[rate_column],
                errors="coerce",
            ),
    })

    if interval_id_column is None:
        result[
            "release_interval_id"
        ] = [
            f"INTERVAL_{number:06d}"
            for number in range(
                1,
                len(result) + 1,
            )
        ]
    else:
        result[
            "release_interval_id"
        ] = frame[
            interval_id_column
        ].astype(str)

    result = result.dropna(
        subset=[
            "site_alias",
            "release_start_utc",
            "release_end_utc",
            "release_rate_kg_h",
        ]
    ).copy()

    # Negative contamination check only needs
    # intervals with a nonzero methane release.
    result = result[
        result[
            "release_rate_kg_h"
        ].gt(0)
    ].copy()

    result = result.sort_values(
        [
            "site_alias",
            "release_start_utc",
            "release_end_utc",
        ]
    ).reset_index(drop=True)

    return result


def load_candidates():
    if not CANDIDATE_INPUT.exists():
        raise FileNotFoundError(
            CANDIDATE_INPUT
        )

    frame = pd.read_csv(
        CANDIDATE_INPUT,
        low_memory=False,
    )

    time_column = find_column(
        frame,
        [
            "candidate_time_parsed_utc",
            "candidate_time_utc",
            "acquisition_time_utc",
        ],
        "Independent candidate table",
    )

    site_column = find_column(
        frame,
        [
            "site_alias",
            "site_name_normalized",
            "site_key",
            "site",
        ],
        "Independent candidate table",
    )

    scene_column = find_column(
        frame,
        [
            "LANDSAT_PRODUCT_ID",
            "candidate_scene_id",
            "scene_id",
            "LANDSAT_SCENE_ID",
            "system:index",
        ],
        "Independent candidate table",
    )

    frame[
        "candidate_acquisition_time_utc"
    ] = pd.to_datetime(
        frame[time_column],
        errors="coerce",
        utc=True,
    )

    frame[
        "candidate_site_alias"
    ] = frame[
        site_column
    ].map(
        normalize_site_alias
    )

    frame[
        "candidate_scene_id_standard"
    ] = frame[
        scene_column
    ].astype(str)

    frame = frame.dropna(
        subset=[
            "candidate_acquisition_time_utc",
            "candidate_site_alias",
            "candidate_scene_id_standard",
        ]
    ).copy()

    if (
        "independent_overpass_key"
        not in frame.columns
    ):
        sensor = (
            frame[
                "landsat_sensor"
            ].astype(str)
            if "landsat_sensor"
            in frame.columns
            else "Landsat"
        )

        frame[
            "independent_overpass_key"
        ] = (
            frame[
                "candidate_site_alias"
            ].astype(str)
            + "|"
            + sensor
            + "|"
            + frame[
                "candidate_acquisition_time_utc"
            ].dt.strftime(
                "%Y-%m-%d"
            )
        )

    if frame[
        "independent_overpass_key"
    ].duplicated().any():
        raise RuntimeError(
            "393 candidate table contains duplicated "
            "independent_overpass_key values."
        )

    return frame


def load_positive_requirements():
    if not POSITIVE_INPUT.exists():
        raise FileNotFoundError(
            POSITIVE_INPUT
        )

    frame = pd.read_csv(
        POSITIVE_INPUT,
        low_memory=False,
    )

    frame["label"] = pd.to_numeric(
        frame["label"],
        errors="raise",
    ).astype(int)

    positive = frame[
        frame["label"].eq(1)
    ].copy()

    site_column = find_column(
        positive,
        [
            "site",
            "site_name",
            "site_name_normalized",
            "site_key",
        ],
        "Confirmed positive table",
    )

    positive[
        "site_alias"
    ] = positive[
        site_column
    ].map(
        normalize_site_alias
    )

    requirements = (
        positive.groupby(
            "site_alias"
        )
        .size()
        .rename(
            "positive_count"
        )
        .reset_index()
    )

    requirements[
        "required_unique_negatives"
    ] = (
        requirements[
            "positive_count"
        ]
        * NEGATIVES_PER_POSITIVE
    )

    return requirements


def audit_candidate(
    candidate,
    release_intervals,
):
    site_alias = candidate[
        "candidate_site_alias"
    ]

    acquisition_time = candidate[
        "candidate_acquisition_time_utc"
    ]

    same_site = release_intervals[
        release_intervals[
            "site_alias"
        ].eq(site_alias)
    ].copy()

    if same_site.empty:
        return {
            "release_interval_count_at_site":
                0,

            "exact_release_overlap":
                pd.NA,

            "overlapping_interval_count":
                0,

            "overlapping_interval_ids":
                "",

            "overlapping_release_rate_max_kg_h":
                np.nan,

            "nearest_nonzero_release_hours":
                np.nan,

            "nearest_release_interval_id":
                pd.NA,

            "clean_exact_overlap":
                False,

            "clean_24h":
                False,

            "release_audit_status":
                "unresolved_no_release_intervals_for_site",
        }

    distances = same_site.apply(
        lambda row:
            interval_distance_hours(
                acquisition_time=
                    acquisition_time,

                start_time=
                    row[
                        "release_start_utc"
                    ],

                end_time=
                    row[
                        "release_end_utc"
                    ],
            ),
        axis=1,
    )

    same_site = same_site.assign(
        distance_hours=distances
    )

    overlap = same_site[
        same_site[
            "distance_hours"
        ].eq(0)
    ].copy()

    nearest_index = (
        same_site[
            "distance_hours"
        ].idxmin()
    )

    nearest = same_site.loc[
        nearest_index
    ]

    nearest_hours = float(
        nearest[
            "distance_hours"
        ]
    )

    exact_overlap = (
        not overlap.empty
    )

    clean_exact = (
        not exact_overlap
    )

    clean_24h = (
        clean_exact
        and nearest_hours
        > SAFETY_BUFFER_HOURS
    )

    if exact_overlap:
        status = (
            "contaminated_exact_release_overlap"
        )

    elif clean_24h:
        status = "clean_more_than_24h"

    else:
        status = (
            "clean_exact_but_within_24h"
        )

    return {
        "release_interval_count_at_site":
            len(same_site),

        "exact_release_overlap":
            exact_overlap,

        "overlapping_interval_count":
            len(overlap),

        "overlapping_interval_ids":
            "|".join(
                overlap[
                    "release_interval_id"
                ].astype(str)
            ),

        "overlapping_release_rate_max_kg_h":
            (
                float(
                    overlap[
                        "release_rate_kg_h"
                    ].max()
                )
                if not overlap.empty
                else np.nan
            ),

        "nearest_nonzero_release_hours":
            nearest_hours,

        "nearest_release_interval_id":
            nearest[
                "release_interval_id"
            ],

        "clean_exact_overlap":
            clean_exact,

        "clean_24h":
            clean_24h,

        "release_audit_status":
            status,
    }


def main():
    candidates = load_candidates()
    intervals = load_release_intervals()
    requirements = (
        load_positive_requirements()
    )

    print("=" * 115)
    print(
        "LANDSAT CANDIDATE RELEASE "
        "CONTAMINATION AUDIT"
    )
    print("=" * 115)

    print(
        "\nIndependent candidates:",
        len(candidates),
    )

    print(
        "Nonzero release intervals:",
        len(intervals),
    )

    audit_rows = []

    for number, candidate in (
        candidates.reset_index(
            drop=True
        ).iterrows()
    ):
        result = audit_candidate(
            candidate,
            intervals,
        )

        audit_rows.append({
            **candidate.to_dict(),
            **result,
        })

        print(
            f"[{number + 1:02d}/{len(candidates)}] "
            f"{candidate['candidate_site_alias']} | "
            f"{candidate['candidate_acquisition_time_utc']} | "
            f"{result['release_audit_status']}",
            flush=True,
        )

    audit = pd.DataFrame(
        audit_rows
    )

    audit.to_csv(
        AUDIT_OUTPUT,
        index=False,
    )

    clean_exact = audit[
        audit[
            "clean_exact_overlap"
        ].eq(True)
    ].copy()

    clean_24h = audit[
        audit[
            "clean_24h"
        ].eq(True)
    ].copy()

    clean_exact.to_csv(
        CLEAN_EXACT_OUTPUT,
        index=False,
    )

    clean_24h.to_csv(
        CLEAN_24H_OUTPUT,
        index=False,
    )

    exact_counts = (
        clean_exact.groupby(
            "candidate_site_alias"
        ).size().rename(
            "clean_exact_candidate_count"
        )
    )

    safe_counts = (
        clean_24h.groupby(
            "candidate_site_alias"
        ).size().rename(
            "clean_24h_candidate_count"
        )
    )

    all_counts = (
        audit.groupby(
            "candidate_site_alias"
        ).size().rename(
            "all_candidate_count"
        )
    )

    site_gap = (
        requirements.merge(
            all_counts,
            left_on="site_alias",
            right_index=True,
            how="left",
        )
        .merge(
            exact_counts,
            left_on="site_alias",
            right_index=True,
            how="left",
        )
        .merge(
            safe_counts,
            left_on="site_alias",
            right_index=True,
            how="left",
        )
    )

    count_columns = [
        "all_candidate_count",
        "clean_exact_candidate_count",
        "clean_24h_candidate_count",
    ]

    for column in count_columns:
        site_gap[column] = (
            site_gap[column]
            .fillna(0)
            .astype(int)
        )

    site_gap[
        "additional_needed_exact"
    ] = (
        site_gap[
            "required_unique_negatives"
        ]
        - site_gap[
            "clean_exact_candidate_count"
        ]
    ).clip(lower=0)

    site_gap[
        "additional_needed_24h"
    ] = (
        site_gap[
            "required_unique_negatives"
        ]
        - site_gap[
            "clean_24h_candidate_count"
        ]
    ).clip(lower=0)

    status_counts = (
        audit[
            "release_audit_status"
        ].value_counts(
            dropna=False
        )
    )

    site_status = pd.crosstab(
        audit[
            "candidate_site_alias"
        ],
        audit[
            "release_audit_status"
        ],
        margins=True,
    )

    report_lines = [
        "=" * 115,
        (
            "LANDSAT CANDIDATE RELEASE "
            "CONTAMINATION AUDIT V1"
        ),
        "=" * 115,
        "",
        (
            f"Independent candidate overpasses: "
            f"{len(audit)}"
        ),
        (
            f"Clean exact-overlap candidates: "
            f"{len(clean_exact)}"
        ),
        (
            f"Clean candidates more than "
            f"{SAFETY_BUFFER_HOURS:.0f}h from release: "
            f"{len(clean_24h)}"
        ),
        "",
        "Release-audit status:",
        status_counts.to_string(),
        "",
        "Status by site:",
        site_status.to_string(),
        "",
        "Site-level negative gap:",
        site_gap.to_string(
            index=False
        ),
        "",
        "Interpretation:",
        (
            "clean_exact_overlap means the Landsat "
            "acquisition was not inside a nonzero "
            "controlled-release interval."
        ),
        (
            "clean_24h additionally requires the "
            "acquisition to be more than 24 hours from "
            "the nearest nonzero release interval."
        ),
        (
            "The 24-hour criterion is the conservative "
            "candidate pool for matched negatives."
        ),
    ]

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 115)
    print("RELEASE AUDIT SUMMARY")
    print("=" * 115)

    print(
        "\nClean exact-overlap candidates:",
        len(clean_exact),
        "/",
        len(audit),
    )

    print(
        "Clean 24h candidates:",
        len(clean_24h),
        "/",
        len(audit),
    )

    print("\nRelease-audit status:")
    print(status_counts)

    print("\nStatus by site:")
    print(site_status)

    print("\nSite-level negative gap:")
    print(
        site_gap.to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(AUDIT_OUTPUT)
    print(CLEAN_EXACT_OUTPUT)
    print(CLEAN_24H_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
