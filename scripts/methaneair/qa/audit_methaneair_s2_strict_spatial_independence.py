from pathlib import Path
from math import radians, sin, cos, asin, sqrt

import pandas as pd


INPUT = Path(
    "outputs/447_methaneair_s2_below500_best_scene_per_event_v1.csv"
)

STRICT_OUTPUT = Path(
    "outputs/449_methaneair_s2_below500_strict_temporal_candidates_v1.csv"
)

PAIR_OUTPUT = Path(
    "outputs/450_methaneair_s2_below500_same_scene_distance_audit_v1.csv"
)

READY_OUTPUT = Path(
    "outputs/451_methaneair_s2_below500_nonoverlap_download_shortlist_v1.csv"
)

REPORT_OUTPUT = Path(
    "outputs/452_methaneair_s2_below500_spatial_audit_report_v1.txt"
)

PATCH_RADIUS_M = 1000
OVERLAP_DISTANCE_M = PATCH_RADIUS_M * 2


def to_boolean(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def haversine_m(lat1, lon1, lat2, lon2):
    earth_radius_m = 6371000.0

    lat1 = radians(float(lat1))
    lon1 = radians(float(lon1))
    lat2 = radians(float(lat2))
    lon2 = radians(float(lon2))

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    value = (
        sin(delta_lat / 2) ** 2
        + cos(lat1)
        * cos(lat2)
        * sin(delta_lon / 2) ** 2
    )

    return (
        2
        * earth_radius_m
        * asin(sqrt(value))
    )


class UnionFind:
    def __init__(self, indices):
        self.parent = {
            index: index
            for index in indices
        }

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[
                self.parent[value]
            ]

            value = self.parent[value]

        return value

    def union(self, first, second):
        first_root = self.find(first)
        second_root = self.find(second)

        if first_root != second_root:
            self.parent[second_root] = first_root


def main():
    frame = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    required = [
        "event_id",
        "emission_kg_hr",
        "scene_id",
        "event_time_utc",
        "acquisition_time_utc",
        "absolute_time_difference_hours",
        "same_utc_date",
        "within_6_hours",
        "latitude",
        "longitude",
        "resolution_status",
    ]

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise KeyError(
            "Missing columns: "
            + ", ".join(missing)
        )

    frame["emission_kg_hr"] = pd.to_numeric(
        frame["emission_kg_hr"],
        errors="coerce",
    )

    frame["latitude"] = pd.to_numeric(
        frame["latitude"],
        errors="coerce",
    )

    frame["longitude"] = pd.to_numeric(
        frame["longitude"],
        errors="coerce",
    )

    frame[
        "absolute_time_difference_hours"
    ] = pd.to_numeric(
        frame[
            "absolute_time_difference_hours"
        ],
        errors="coerce",
    )

    same_date = to_boolean(
        frame["same_utc_date"]
    )

    within_six_hours = to_boolean(
        frame["within_6_hours"]
    )

    strict = frame[
        frame["resolution_status"].eq(
            "best_candidate_resolved"
        )
        & frame["emission_kg_hr"].gt(0)
        & frame["emission_kg_hr"].lt(500)
        & same_date
        & within_six_hours
        & frame[
            "absolute_time_difference_hours"
        ].le(6)
    ].copy()

    strict = strict.dropna(
        subset=[
            "event_id",
            "scene_id",
            "latitude",
            "longitude",
        ]
    )

    strict = strict.drop_duplicates(
        subset=["event_id"],
        keep="first",
    ).reset_index(drop=True)

    strict[
        "scene_event_count"
    ] = strict.groupby(
        "scene_id"
    )["event_id"].transform("size")

    pair_records = []
    cluster_assignments = {}

    for scene_number, (
        scene_id,
        scene_group,
    ) in enumerate(
        strict.groupby(
            "scene_id",
            sort=True,
        ),
        start=1,
    ):
        indices = list(scene_group.index)

        union_find = UnionFind(indices)

        for first_position in range(
            len(indices)
        ):
            for second_position in range(
                first_position + 1,
                len(indices),
            ):
                first_index = indices[
                    first_position
                ]

                second_index = indices[
                    second_position
                ]

                first = strict.loc[
                    first_index
                ]

                second = strict.loc[
                    second_index
                ]

                distance_m = haversine_m(
                    first["latitude"],
                    first["longitude"],
                    second["latitude"],
                    second["longitude"],
                )

                patch_overlap = (
                    distance_m
                    < OVERLAP_DISTANCE_M
                )

                if patch_overlap:
                    union_find.union(
                        first_index,
                        second_index,
                    )

                pair_records.append({
                    "scene_id":
                        scene_id,

                    "event_id_1":
                        first["event_id"],

                    "event_id_2":
                        second["event_id"],

                    "emission_kg_hr_1":
                        first["emission_kg_hr"],

                    "emission_kg_hr_2":
                        second["emission_kg_hr"],

                    "distance_m":
                        distance_m,

                    "patch_radius_m":
                        PATCH_RADIUS_M,

                    "patch_overlap_threshold_m":
                        OVERLAP_DISTANCE_M,

                    "patches_overlap":
                        patch_overlap,
                })

        roots = {}

        for index in indices:
            root = union_find.find(index)

            if root not in roots:
                roots[root] = len(roots) + 1

            cluster_number = roots[root]

            cluster_assignments[index] = (
                f"S2SCENE_{scene_number:03d}_"
                f"SOURCE_{cluster_number:02d}"
            )

    strict[
        "scene_source_cluster_id"
    ] = strict.index.map(
        cluster_assignments
    )

    strict[
        "spatial_cluster_event_count"
    ] = strict.groupby(
        "scene_source_cluster_id"
    )["event_id"].transform("size")

    strict[
        "patch_overlap_risk"
    ] = strict[
        "spatial_cluster_event_count"
    ].gt(1)

    strict[
        "ready_for_direct_download"
    ] = ~strict[
        "patch_overlap_risk"
    ]

    strict[
        "evaluation_scene_group"
    ] = strict["scene_id"]

    strict = strict.sort_values(
        [
            "patch_overlap_risk",
            "emission_kg_hr",
            "scene_id",
            "event_id",
        ]
    ).reset_index(drop=True)

    strict.to_csv(
        STRICT_OUTPUT,
        index=False,
    )

    pairs = pd.DataFrame(
        pair_records
    )

    if pairs.empty:
        pairs = pd.DataFrame(columns=[
            "scene_id",
            "event_id_1",
            "event_id_2",
            "emission_kg_hr_1",
            "emission_kg_hr_2",
            "distance_m",
            "patch_radius_m",
            "patch_overlap_threshold_m",
            "patches_overlap",
        ])

    pairs.to_csv(
        PAIR_OUTPUT,
        index=False,
    )

    ready = strict[
        strict[
            "ready_for_direct_download"
        ]
    ].copy()

    ready.to_csv(
        READY_OUTPUT,
        index=False,
    )

    strict_event_count = len(strict)

    unique_scene_count = strict[
        "scene_id"
    ].nunique()

    shared_scene_count = int(
        strict.groupby(
            "scene_id"
        ).size().gt(1).sum()
    )

    overlapping_pair_count = int(
        pairs[
            "patches_overlap"
        ].sum()
    ) if not pairs.empty else 0

    overlap_event_count = int(
        strict[
            "patch_overlap_risk"
        ].sum()
    )

    independent_source_units = strict[
        "scene_source_cluster_id"
    ].nunique()

    report_lines = [
        "=" * 105,
        "METHANEAIR–S2 BELOW-500 KG/H SPATIAL AUDIT V1",
        "=" * 105,
        "",
        (
            "Strict temporal definition: "
            "same UTC date and within 6 hours"
        ),
        (
            f"Patch overlap definition: "
            f"source distance < {OVERLAP_DISTANCE_M} m"
        ),
        "",
        (
            "Strict temporal candidate events: "
            f"{strict_event_count}"
        ),
        (
            "Unique Sentinel-2 scenes: "
            f"{unique_scene_count}"
        ),
        (
            "Scenes containing multiple candidate events: "
            f"{shared_scene_count}"
        ),
        (
            "Independent scene-source units: "
            f"{independent_source_units}"
        ),
        (
            "Overlapping event pairs: "
            f"{overlapping_pair_count}"
        ),
        (
            "Events requiring overlap review: "
            f"{overlap_event_count}"
        ),
        (
            "Non-overlapping events ready for download: "
            f"{len(ready)}"
        ),
        "",
        "Emission-bin summary:",
        pd.cut(
            strict["emission_kg_hr"],
            bins=[0, 200, 500],
            labels=[
                "0_to_200",
                "200_to_500",
            ],
            right=False,
        )
        .value_counts()
        .sort_index()
        .to_string(),
    ]

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("=" * 105)
    print(
        "METHANEAIR–S2 STRICT SPATIAL AUDIT"
    )
    print("=" * 105)

    print(
        "\nStrict temporal candidate events:",
        strict_event_count,
    )

    print(
        "Unique Sentinel-2 scenes:",
        unique_scene_count,
    )

    print(
        "Scenes with multiple events:",
        shared_scene_count,
    )

    print(
        "Independent scene-source units:",
        independent_source_units,
    )

    print(
        "Overlapping event pairs:",
        overlapping_pair_count,
    )

    print(
        "Events requiring overlap review:",
        overlap_event_count,
    )

    print(
        "Non-overlapping events ready for download:",
        len(ready),
    )

    print("\nSaved:")
    print(STRICT_OUTPUT)
    print(PAIR_OUTPUT)
    print(READY_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
