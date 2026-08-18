from pathlib import Path
import numpy as np
import pandas as pd


INPUT = Path(
    "outputs/301_s2_strict_review_compact.csv"
)

FULL_OUTPUT = Path(
    "outputs/307_s2_direct_strict_ground_truth_v1.csv"
)

PRIMARY_OUTPUT = Path(
    "outputs/308_s2_direct_primary_benchmark_v1.csv"
)


DECISIONS = {
    1: {
        "review_status": "conflicting_very_low_flow",
        "strict_label": 1,
        "primary_include": False,
        "preferred_release_rate_kg_h": 4.9500564248826,
        "release_start_utc": "2022-11-28 18:20:01+00:00",
        "release_end_utc": "2022-11-28 18:25:01+00:00",
        "rate_source": "exact_window_ch4_kgh_mean",
        "ground_truth_conflict": True,
        "review_notes": (
            "Meter-derived CH4 flow is about 4.95 kg/h, "
            "but FacilityEmissionRate is zero. Retained as "
            "an exploratory sensitivity case only."
        ),
    },

    3: {
        "review_status": "strict_negative_zero_release",
        "strict_label": 0,
        "primary_include": True,
        "preferred_release_rate_kg_h": 0.0,
        "release_start_utc": "",
        "release_end_utc": "",
        "rate_source": "facility_emission_and_source_count_zero",
        "ground_truth_conflict": False,
        "review_notes": (
            "FacilityEmissionRate and reported source count "
            "indicate no release."
        ),
    },

    6: {
        "review_status": "strict_positive",
        "strict_label": 1,
        "primary_include": True,
        "preferred_release_rate_kg_h": 1428.3903451392,
        "release_start_utc": "2022-11-18 18:20:03+00:00",
        "release_end_utc": "2022-11-18 18:25:03+00:00",
        "rate_source": "exact_window_ch4_kgh_mean",
        "ground_truth_conflict": False,
        "review_notes": (
            "S2 acquisition coincides with the end of a "
            "five-minute nonzero controlled-release interval."
        ),
    },

    8: {
        "review_status": "strict_positive",
        "strict_label": 1,
        "primary_include": True,
        "preferred_release_rate_kg_h": 1495.75868707644,
        "release_start_utc": "2022-11-15 18:10:08+00:00",
        "release_end_utc": "2022-11-15 18:15:08+00:00",
        "rate_source": "exact_window_ch4_kgh_mean",
        "ground_truth_conflict": False,
        "review_notes": (
            "S2 acquisition coincides with the end of a "
            "five-minute nonzero controlled-release interval."
        ),
    },

    12: {
        "review_status": "strict_positive",
        "strict_label": 1,
        "primary_include": True,
        "preferred_release_rate_kg_h": 1190.059321779225,
        "release_start_utc": "2022-11-08 18:20:05+00:00",
        "release_end_utc": "2022-11-08 18:25:05+00:00",
        "rate_source": "exact_window_ch4_kgh_mean",
        "ground_truth_conflict": False,
        "review_notes": (
            "S2 acquisition coincides with the end of a "
            "five-minute nonzero controlled-release interval."
        ),
    },

    13: {
        "review_status": "strict_positive",
        "strict_label": 1,
        "primary_include": True,
        "preferred_release_rate_kg_h": 1386.18020849443,
        "release_start_utc": "2021-11-03 18:12:18+00:00",
        "release_end_utc": "",
        "rate_source": "cr_kgh_CH4_mean300",
        "ground_truth_conflict": False,
        "review_notes": (
            "Release was active before overpass; rolling "
            "meter averages are nonzero at acquisition."
        ),
    },

    14: {
        "review_status": "strict_positive",
        "strict_label": 1,
        "primary_include": True,
        "preferred_release_rate_kg_h": 1675.6437801655145,
        "release_start_utc": "2021-10-22 18:16:57+00:00",
        "release_end_utc": "",
        "rate_source": "cr_kgh_CH4_mean300",
        "ground_truth_conflict": False,
        "review_notes": (
            "Release was active before overpass; rolling "
            "meter averages are nonzero at acquisition."
        ),
    },

    15: {
        "review_status": "strict_positive",
        "strict_label": 1,
        "primary_include": True,
        "preferred_release_rate_kg_h": 5024.149730095228,
        "release_start_utc": "2021-10-29 18:09:15+00:00",
        "release_end_utc": "",
        "rate_source": "cr_kgh_CH4_mean300",
        "ground_truth_conflict": False,
        "review_notes": (
            "Meter-derived rolling CH4 flow is strongly "
            "positive at overpass."
        ),
    },

    16: {
        "review_status": "strict_positive",
        "strict_label": 1,
        "primary_include": True,
        "preferred_release_rate_kg_h": 3494.031668919704,
        "release_start_utc": "2021-10-27 18:13:45+00:00",
        "release_end_utc": "",
        "rate_source": "cr_kgh_CH4_mean300",
        "ground_truth_conflict": False,
        "review_notes": (
            "Release was active before overpass; rolling "
            "meter averages are nonzero at acquisition."
        ),
    },

    17: {
        "review_status": "strict_positive",
        "strict_label": 1,
        "primary_include": True,
        "preferred_release_rate_kg_h": 7175.659697032253,
        "release_start_utc": "2021-10-19 18:05:36+00:00",
        "release_end_utc": "",
        "rate_source": "cr_kgh_CH4_mean300",
        "ground_truth_conflict": False,
        "review_notes": (
            "Release was active before overpass; rolling "
            "meter averages are nonzero at acquisition."
        ),
    },
}


def emission_bin(rate):
    if pd.isna(rate):
        return "missing"

    if rate == 0:
        return "0_negative"

    if rate < 200:
        return "0_to_200"

    if rate < 500:
        return "200_to_500"

    if rate < 1000:
        return "500_to_1000"

    if rate < 2000:
        return "1000_to_2000"

    return "2000_plus"


def main():
    df = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    direct = df[
        df["candidate_rank"].isin(
            DECISIONS.keys()
        )
    ].copy()

    if len(direct) != len(DECISIONS):
        raise RuntimeError(
            f"Expected {len(DECISIONS)} direct rows, "
            f"found {len(direct)}."
        )

    decision_frame = pd.DataFrame([
        {
            "candidate_rank": rank,
            **decision,
        }
        for rank, decision in DECISIONS.items()
    ])

    result = direct.merge(
        decision_frame,
        on="candidate_rank",
        how="left",
        validate="one_to_one",
        suffixes=("", "_final"),
    )

    result[
        "preferred_release_rate_kg_h"
    ] = pd.to_numeric(
        result[
            "preferred_release_rate_kg_h"
        ],
        errors="coerce",
    )

    result["final_emission_bin"] = (
        result[
            "preferred_release_rate_kg_h"
        ].apply(emission_bin)
    )

    result["low_emission_lt1000"] = (
        result[
            "preferred_release_rate_kg_h"
        ].gt(0)
        & result[
            "preferred_release_rate_kg_h"
        ].lt(1000)
    )

    result["medium_emission_1000_2000"] = (
        result[
            "preferred_release_rate_kg_h"
        ].ge(1000)
        & result[
            "preferred_release_rate_kg_h"
        ].lt(2000)
    )

    result["ground_truth_version"] = (
        "s2_direct_strict_v1_meter_preferred"
    )

    result = result.sort_values(
        "candidate_rank"
    ).reset_index(drop=True)

    primary = result[
        result["primary_include"].eq(True)
    ].copy()

    result.to_csv(
        FULL_OUTPUT,
        index=False,
    )

    primary.to_csv(
        PRIMARY_OUTPUT,
        index=False,
    )

    print("=" * 105)
    print("SENTINEL-2 DIRECT STRICT GROUND TRUTH")
    print("=" * 105)

    print("\nFull reviewed rows:", len(result))
    print("Primary benchmark rows:", len(primary))

    print("\nPrimary labels:")
    print(
        primary["strict_label"]
        .value_counts()
        .sort_index()
    )

    print("\nPrimary emission bins:")
    print(
        primary["final_emission_bin"]
        .value_counts()
        .sort_index()
    )

    print(
        "\nPrimary low-emission positives <1000 kg/h:",
        int(
            (
                primary["strict_label"].eq(1)
                & primary["low_emission_lt1000"]
            ).sum()
        ),
    )

    print(
        "Exploratory conflicting low-flow cases:",
        int(
            result[
                "review_status"
            ].eq(
                "conflicting_very_low_flow"
            ).sum()
        ),
    )

    print("\nFinal reviewed cases:")
    print(
        result[
            [
                "candidate_rank",
                "event_id",
                "review_status",
                "strict_label",
                "primary_include",
                "preferred_release_rate_kg_h",
                "final_emission_bin",
                "rate_source",
            ]
        ].to_string(
            index=False,
        )
    )

    print("\nSaved:")
    print(FULL_OUTPUT)
    print(PRIMARY_OUTPUT)


if __name__ == "__main__":
    main()
