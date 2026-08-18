from pathlib import Path

import pandas as pd


GROUND_TRUTH_INPUT = Path(
    "outputs/307_s2_direct_strict_ground_truth_v1.csv"
)

RELEASE_INVENTORY_INPUT = Path(
    "outputs/309_all_exact_release_intervals_for_s2.csv"
)

INVENTORY_OUTPUT = Path(
    "outputs/349_s2_high_emission_positive_inventory_v1.csv"
)

AUDIT_OUTPUT = Path(
    "outputs/350_s2_high_emission_positive_inventory_audit.txt"
)


HIGH_EMISSION_THRESHOLD_KG_H = 1000.0


def parse_bool(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin([
            "true",
            "1",
            "yes",
            "include",
            "included",
        ])
    )


def find_column(
    frame,
    candidates,
    required=False,
):
    for column in candidates:
        if column in frame.columns:
            return column

    if required:
        raise KeyError(
            "找不到任何候選欄位："
            + ", ".join(candidates)
        )

    return None


def normalized_site(series):
    return (
        series.astype(str)
        .str.strip()
        .str.replace(
            r"\s+",
            "_",
            regex=True,
        )
    )


def load_site_coordinates():
    empty = pd.DataFrame(
        columns=[
            "site_key",
            "lat",
            "lon",
        ]
    )

    if not RELEASE_INVENTORY_INPUT.exists():
        print(
            "Warning: 找不到場址座標檔：",
            RELEASE_INVENTORY_INPUT,
        )

        return empty

    release = pd.read_csv(
        RELEASE_INVENTORY_INPUT,
        low_memory=False,
    )

    site_column = find_column(
        release,
        [
            "site",
            "site_name",
            "release_site",
        ],
    )

    lat_column = find_column(
        release,
        [
            "lat",
            "latitude",
            "source_lat",
            "release_lat",
            "site_lat",
            "release_latitude",
        ],
    )

    lon_column = find_column(
        release,
        [
            "lon",
            "longitude",
            "source_lon",
            "release_lon",
            "site_lon",
            "release_longitude",
        ],
    )

    if (
        site_column is None
        or lat_column is None
        or lon_column is None
    ):
        print(
            "Warning: 309 檔案沒有找到完整的 "
            "site/lat/lon 欄位。"
        )

        print(
            "309 columns:",
            list(release.columns),
        )

        return empty

    coordinates = pd.DataFrame({
        "site_key":
            normalized_site(
                release[site_column]
            ),

        "lat":
            pd.to_numeric(
                release[lat_column],
                errors="coerce",
            ),

        "lon":
            pd.to_numeric(
                release[lon_column],
                errors="coerce",
            ),
    })

    coordinates = (
        coordinates.dropna(
            subset=[
                "site_key",
                "lat",
                "lon",
            ]
        )
        .groupby(
            "site_key",
            as_index=False,
        )
        .agg({
            "lat": "median",
            "lon": "median",
        })
    )

    return coordinates


def main():
    if not GROUND_TRUTH_INPUT.exists():
        raise FileNotFoundError(
            GROUND_TRUTH_INPUT
        )

    raw = pd.read_csv(
        GROUND_TRUTH_INPUT,
        low_memory=False,
    )

    required_columns = [
        "event_id",
        "filename",
        "site_name",
        "actual_s2_time",
        "strict_label",
        "primary_include",
        "preferred_release_rate_kg_h",
    ]

    missing = [
        column
        for column in required_columns
        if column not in raw.columns
    ]

    if missing:
        raise KeyError(
            "307 檔案缺少欄位："
            + ", ".join(missing)
        )

    raw[
        "actual_s2_time"
    ] = pd.to_datetime(
        raw["actual_s2_time"],
        errors="coerce",
        utc=True,
    )

    raw[
        "preferred_release_rate_kg_h"
    ] = pd.to_numeric(
        raw[
            "preferred_release_rate_kg_h"
        ],
        errors="coerce",
    )

    raw[
        "strict_label_numeric"
    ] = pd.to_numeric(
        raw["strict_label"],
        errors="coerce",
    )

    raw[
        "primary_include_bool"
    ] = parse_bool(
        raw["primary_include"]
    )

    selection = (
        raw[
            "strict_label_numeric"
        ].eq(1)
        & raw[
            "primary_include_bool"
        ]
        & raw[
            "preferred_release_rate_kg_h"
        ].ge(
            HIGH_EMISSION_THRESHOLD_KG_H
        )
        & raw[
            "actual_s2_time"
        ].notna()
    )

    selected = raw[
        selection
    ].copy()

    selected["site_key"] = (
        normalized_site(
            selected["site_name"]
        )
    )

    selected = (
        selected.sort_values(
            [
                "actual_s2_time",
                "preferred_release_rate_kg_h",
            ]
        )
        .drop_duplicates(
            subset=[
                "site_key",
                "actual_s2_time",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    coordinates = (
        load_site_coordinates()
    )

    selected = selected.merge(
        coordinates,
        on="site_key",
        how="left",
        validate="many_to_one",
    )

    selected[
        "positive_id"
    ] = [
        f"S2_HIGH_POS_{number:02d}"
        for number in range(
            1,
            len(selected) + 1,
        )
    ]

    selected["label"] = 1

    selected[
        "dataset_role"
    ] = (
        "strict_high_emission_positive"
    )

    # 307 裡沒有 Earth Engine scene ID。
    # 下一步再根據 site + actual_s2_time 精確解析。
    selected["scene_id"] = pd.NA

    selected[
        "scene_resolution_status"
    ] = "pending_earth_engine_resolution"

    selected[
        "scene_lookup_time_utc"
    ] = selected[
        "actual_s2_time"
    ]

    selected[
        "high_emission_threshold_kg_h"
    ] = (
        HIGH_EMISSION_THRESHOLD_KG_H
    )

    selected[
        "inventory_version"
    ] = (
        "s2_high_emission_positive_v1"
    )

    optional_columns = [
        "candidate_rank",
        "event_id",
        "filename",
        "site_name",
        "actual_s2_time",
        "original_event_time",
        "s2_event_difference_minutes",
        "preferred_release_rate_kg_h",
        "release_start_utc",
        "release_end_utc",
        "rate_source",
        "raw_row_index",
        "raw_source_file",
        "raw_source_sheet",
        "review_status_final",
        "ground_truth_conflict",
        "review_notes_final",
        "final_emission_bin",
    ]

    existing_optional = [
        column
        for column in optional_columns
        if column in selected.columns
    ]

    final_columns = [
        "positive_id",
        "scene_id",
        "scene_resolution_status",
        "scene_lookup_time_utc",
        *existing_optional,
        "site_key",
        "lat",
        "lon",
        "label",
        "dataset_role",
        "high_emission_threshold_kg_h",
        "inventory_version",
    ]

    inventory = selected[
        final_columns
    ].copy()

    inventory.to_csv(
        INVENTORY_OUTPUT,
        index=False,
    )

    missing_coordinates = int(
        inventory[
            ["lat", "lon"]
        ].isna().any(axis=1).sum()
    )

    report_lines = [
        "=" * 110,
        (
            "SENTINEL-2 HIGH-EMISSION "
            "POSITIVE INVENTORY AUDIT"
        ),
        "=" * 110,
        "",
        f"Input rows: {len(raw)}",
        (
            "Strict positive rows: "
            f"{int(raw['strict_label_numeric'].eq(1).sum())}"
        ),
        (
            "Strict primary included rows: "
            f"{int((raw['strict_label_numeric'].eq(1) & raw['primary_include_bool']).sum())}"
        ),
        (
            "High-emission rows before deduplication: "
            f"{int(selection.sum())}"
        ),
        (
            "Unique high-emission positive scenes: "
            f"{len(inventory)}"
        ),
        (
            "Rows missing site coordinates: "
            f"{missing_coordinates}"
        ),
        "",
        "Important:",
        (
            "The ground-truth table does not contain "
            "Earth Engine scene IDs. Exact scene IDs "
            "must be resolved in the next step using "
            "site coordinates and actual_s2_time."
        ),
    ]

    if not inventory.empty:
        report_lines.extend([
            "",
            "Release-rate statistics:",
            inventory[
                "preferred_release_rate_kg_h"
            ].describe().to_string(),
            "",
            "Scenes per site:",
            inventory[
                "site_name"
            ].value_counts().to_string(),
            "",
            "High-emission inventory:",
            inventory[
                [
                    "positive_id",
                    "event_id",
                    "site_name",
                    "actual_s2_time",
                    "preferred_release_rate_kg_h",
                    "lat",
                    "lon",
                    "scene_resolution_status",
                ]
            ].to_string(index=False),
        ])

    AUDIT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("=" * 110)
    print(
        "SENTINEL-2 HIGH-EMISSION "
        "POSITIVE INVENTORY"
    )
    print("=" * 110)

    print(
        "\nInput rows:",
        len(raw),
    )

    print(
        "Strict included positive rows:",
        int(
            (
                raw[
                    "strict_label_numeric"
                ].eq(1)
                & raw[
                    "primary_include_bool"
                ]
            ).sum()
        ),
    )

    print(
        "Unique high-emission positive scenes:",
        len(inventory),
    )

    print(
        "Rows missing site coordinates:",
        missing_coordinates,
    )

    if not inventory.empty:
        print(
            "\nRelease-rate statistics:"
        )

        print(
            inventory[
                "preferred_release_rate_kg_h"
            ].describe()
        )

        print(
            "\nScenes per site:"
        )

        print(
            inventory[
                "site_name"
            ].value_counts()
        )

        print(
            "\nHigh-emission positive inventory:"
        )

        print(
            inventory[
                [
                    "positive_id",
                    "event_id",
                    "site_name",
                    "actual_s2_time",
                    "preferred_release_rate_kg_h",
                    "lat",
                    "lon",
                    "scene_resolution_status",
                ]
            ].to_string(
                index=False
            )
        )

    print("\nSaved:")
    print(INVENTORY_OUTPUT)
    print(AUDIT_OUTPUT)


if __name__ == "__main__":
    main()
