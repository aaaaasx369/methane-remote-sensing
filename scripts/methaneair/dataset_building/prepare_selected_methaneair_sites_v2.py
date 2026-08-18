#!/usr/bin/env python3
"""
Prepare selected MethaneAIR sites using fixed candidate-site coordinates.

Why v2 exists
-------------
DBSCAN cluster labels are not stable if the input rows or filtering change.
This script does NOT recompute cluster IDs. Instead, it reads the exact site
centroids from:
    outputs/510_methaneair_candidate_sites.csv

It then assigns events to the selected centroids by haversine distance.

Default selected sites:
    MethaneAIR_site_073
    MethaneAIR_site_102
    MethaneAIR_site_120

Outputs:
    outputs/511_selected_methaneair_site_events_v2.csv
    outputs/512_selected_methaneair_positive_manifest_v2.csv
    outputs/513_selected_methaneair_positive_audit_v2.csv
    outputs/514_selected_methaneair_unmatched_patch_rows_v2.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0088

EVENT_ID_ALIASES = ("event_id", "sample_id", "plume_id", "id")
LAT_ALIASES = ("lat", "latitude", "source_latitude", "source_lat")
LON_ALIASES = ("lon", "longitude", "lng", "source_longitude", "source_lon")
TIME_ALIASES = (
    "datetime_utc", "event_time_utc", "acquisition_time_utc",
    "timestamp_utc", "datetime",
)
EMISSION_ALIASES = (
    "emission_kg_hr", "emission_kg_h", "release_rate_kg_h",
    "emission_rate_kg_h",
)
PATCH_EVENT_ALIASES = ("event_id", "sample_id", "plume_id")
PATCH_PATH_ALIASES = (
    "patch_path", "relative_path", "file_path", "filepath",
    "image_path", "filename", "resolved_patch_path",
)
SCENE_ALIASES = (
    "scene_id", "s2_scene_id", "system_index", "image_id",
    "product_id", "system:index",
)
PATCH_TIME_ALIASES = (
    "acquisition_time_utc", "s2_time_utc", "scene_time_utc",
    "datetime_utc", "acquisition_datetime",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare selected MethaneAIR sites using fixed site centroids."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/Users/happydoraaa/methane_release_project"),
    )
    parser.add_argument(
        "--selected-sites",
        nargs="+",
        default=[
            "MethaneAIR_site_073",
            "MethaneAIR_site_102",
            "MethaneAIR_site_120",
        ],
    )
    parser.add_argument(
        "--assignment-radius-km",
        type=float,
        default=1.0,
        help="Maximum event-to-site-centroid distance.",
    )
    return parser.parse_args()


def first_column(df: pd.DataFrame, aliases: Iterable[str]) -> Optional[str]:
    lower = {str(column).strip().lower(): column for column in df.columns}
    for alias in aliases:
        if alias.lower() in lower:
            return lower[alias.lower()]
    return None


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    values = [lat1, lon1, lat2, lon2]
    if any(pd.isna(value) for value in values):
        return np.nan

    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    delta_phi = math.radians(float(lat2) - float(lat1))
    delta_lambda = math.radians(float(lon2) - float(lon1))

    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(delta_lambda / 2) ** 2
    )

    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def normalize_identifier(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def resolve_existing_path(root: Path, value: object) -> str:
    if value is None or pd.isna(value):
        return ""

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return ""

    raw = Path(text).expanduser()
    candidates = [
        raw,
        root / raw,
        root / "outputs" / raw,
        root / "patches" / raw,
        root / "images" / raw,
        root / "data" / raw,
        root / "downloads" / raw,
        root / "outputs" / raw.name,
        root / "patches" / raw.name,
        root / "images" / raw.name,
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return str(candidate.resolve())

    for folder_name in ("patches", "images", "data", "downloads", "outputs"):
        folder = root / folder_name
        if not folder.exists():
            continue

        matches = list(folder.rglob(raw.name))
        if len(matches) == 1:
            return str(matches[0].resolve())

    return ""


def infer_scene_id(path_value: object, fallback: str) -> str:
    text = str(path_value).strip()
    if text and text.lower() not in {"nan", "none", "<na>"}:
        return Path(text).stem
    return fallback


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    outputs = root / "outputs"

    candidate_file = outputs / "510_methaneair_candidate_sites.csv"
    event_file = outputs / "15_methaneair_s2_landsat_availability.csv"
    patch_file = outputs / "18_methaneair_s2_dataset_table.csv"

    for path in (candidate_file, event_file, patch_file):
        if not path.exists():
            raise SystemExit(f"Required file not found: {path}")

    candidate_sites = pd.read_csv(candidate_file)
    events = pd.read_csv(event_file)
    patches = pd.read_csv(patch_file)

    required_site_columns = {
        "candidate_site_id", "latitude", "longitude"
    }
    missing_site_columns = required_site_columns - set(candidate_sites.columns)
    if missing_site_columns:
        raise SystemExit(
            "Candidate-site table is missing: "
            + ", ".join(sorted(missing_site_columns))
        )

    selected_sites = candidate_sites[
        candidate_sites["candidate_site_id"].astype(str).isin(args.selected_sites)
    ].copy()

    missing_requested = set(args.selected_sites) - set(
        selected_sites["candidate_site_id"].astype(str)
    )
    if missing_requested:
        raise SystemExit(
            "Selected site IDs not found in 510 table: "
            + ", ".join(sorted(missing_requested))
        )

    event_id_col = first_column(events, EVENT_ID_ALIASES)
    lat_col = first_column(events, LAT_ALIASES)
    lon_col = first_column(events, LON_ALIASES)
    time_col = first_column(events, TIME_ALIASES)
    emission_col = first_column(events, EMISSION_ALIASES)

    missing_event_fields = [
        name
        for name, column in {
            "event ID": event_id_col,
            "latitude": lat_col,
            "longitude": lon_col,
            "time": time_col,
            "emission": emission_col,
        }.items()
        if column is None
    ]
    if missing_event_fields:
        raise SystemExit(
            "Event table is missing: "
            + ", ".join(missing_event_fields)
            + f"\nAvailable columns: {list(events.columns)}"
        )

    work = events.copy()
    work["event_id_canonical"] = normalize_identifier(work[event_id_col])
    work["latitude_canonical"] = pd.to_numeric(work[lat_col], errors="coerce")
    work["longitude_canonical"] = pd.to_numeric(work[lon_col], errors="coerce")
    work["datetime_utc_canonical"] = pd.to_datetime(
        work[time_col], errors="coerce", utc=True
    )
    work["emission_kg_h_canonical"] = pd.to_numeric(
        work[emission_col], errors="coerce"
    )

    work = work.dropna(
        subset=[
            "event_id_canonical",
            "latitude_canonical",
            "longitude_canonical",
            "datetime_utc_canonical",
        ]
    ).copy()

    # Reproduce the same availability filtering used by the ranking script.
    s2_count_columns = [
        column
        for column in ("s2_count_pm1day", "s2_count")
        if column in work.columns
    ]
    if s2_count_columns:
        availability = np.zeros(len(work), dtype=bool)
        for column in s2_count_columns:
            availability |= (
                pd.to_numeric(work[column], errors="coerce")
                .fillna(0)
                .to_numpy()
                > 0
            )
        work = work[availability].copy()

    # Assign each event to its nearest selected fixed centroid.
    assignments = []
    for _, event in work.iterrows():
        distances = []
        for _, site in selected_sites.iterrows():
            distance = haversine_km(
                event["latitude_canonical"],
                event["longitude_canonical"],
                site["latitude"],
                site["longitude"],
            )
            distances.append(
                (
                    str(site["candidate_site_id"]),
                    float(site["latitude"]),
                    float(site["longitude"]),
                    distance,
                )
            )

        distances.sort(key=lambda item: item[3])
        site_id, site_lat, site_lon, distance = distances[0]

        if distance <= args.assignment_radius_km:
            record = event.to_dict()
            record["site_id"] = site_id
            record["site_centroid_latitude"] = site_lat
            record["site_centroid_longitude"] = site_lon
            record["distance_to_site_centroid_km"] = distance
            assignments.append(record)

    selected_events = pd.DataFrame(assignments)

    if selected_events.empty:
        raise SystemExit(
            "No events matched the selected fixed centroids within "
            f"{args.assignment_radius_km:g} km."
        )

    selected_events = (
        selected_events.sort_values(["site_id", "datetime_utc_canonical"])
        .drop_duplicates(subset=["site_id", "event_id_canonical"], keep="first")
        .reset_index(drop=True)
    )

    selected_events_path = (
        outputs / "511_selected_methaneair_site_events_v2.csv"
    )
    selected_events.to_csv(selected_events_path, index=False)

    patch_event_col = first_column(patches, PATCH_EVENT_ALIASES)
    patch_path_col = first_column(patches, PATCH_PATH_ALIASES)
    scene_col = first_column(patches, SCENE_ALIASES)
    patch_time_col = first_column(patches, PATCH_TIME_ALIASES)

    if patch_event_col is None:
        raise SystemExit(
            "Patch table has no event ID column.\n"
            f"Available columns: {list(patches.columns)}"
        )

    patch_work = patches.copy()
    patch_work["event_id_canonical"] = normalize_identifier(
        patch_work[patch_event_col]
    )

    patch_work["patch_path_raw"] = (
        patch_work[patch_path_col] if patch_path_col else ""
    )
    patch_work["resolved_patch_path"] = patch_work["patch_path_raw"].map(
        lambda value: resolve_existing_path(root, value)
    )

    if scene_col:
        patch_work["scene_id"] = normalize_identifier(patch_work[scene_col])
    else:
        patch_work["scene_id"] = ""

    missing_scene = patch_work["scene_id"].isin(
        ["", "nan", "None", "<NA>"]
    )
    patch_work.loc[missing_scene, "scene_id"] = patch_work.loc[
        missing_scene
    ].apply(
        lambda row: infer_scene_id(
            row["resolved_patch_path"] or row["patch_path_raw"],
            f"event_{row['event_id_canonical']}",
        ),
        axis=1,
    )

    if patch_time_col:
        patch_work["s2_acquisition_time_utc"] = pd.to_datetime(
            patch_work[patch_time_col], errors="coerce", utc=True
        )
    else:
        patch_work["s2_acquisition_time_utc"] = pd.NaT

    patch_keep = [
        "event_id_canonical",
        "scene_id",
        "patch_path_raw",
        "resolved_patch_path",
        "s2_acquisition_time_utc",
    ]

    positive_manifest = selected_events.merge(
        patch_work[patch_keep],
        on="event_id_canonical",
        how="left",
        validate="one_to_many",
    )

    positive_manifest["label"] = 1
    positive_manifest["source_origin"] = "MethaneAIR"
    positive_manifest["ground_truth_type"] = "observational_plume"
    positive_manifest["patch_exists_now"] = (
        positive_manifest["resolved_patch_path"].astype(str).str.len() > 0
    )

    positive_manifest_path = (
        outputs / "512_selected_methaneair_positive_manifest_v2.csv"
    )
    positive_manifest.to_csv(positive_manifest_path, index=False)

    unmatched = positive_manifest[
        ~positive_manifest["patch_exists_now"]
    ].copy()
    unmatched_path = outputs / "514_selected_methaneair_unmatched_patch_rows_v2.csv"
    unmatched.to_csv(unmatched_path, index=False)

    audit = (
        positive_manifest.groupby("site_id", dropna=False)
        .agg(
            event_rows=("event_id_canonical", "nunique"),
            manifest_rows=("event_id_canonical", "size"),
            patch_rows=("patch_exists_now", "sum"),
            unique_scenes=(
                "scene_id",
                lambda values: values.astype(str)
                .replace(
                    {
                        "": np.nan,
                        "nan": np.nan,
                        "None": np.nan,
                        "<NA>": np.nan,
                    }
                )
                .nunique(),
            ),
            unique_event_dates=(
                "datetime_utc_canonical",
                lambda values: pd.to_datetime(
                    values, errors="coerce", utc=True
                ).dt.date.nunique(),
            ),
            unique_s2_dates=(
                "s2_acquisition_time_utc",
                lambda values: pd.to_datetime(
                    values, errors="coerce", utc=True
                ).dt.date.nunique(),
            ),
            minimum_emission_kg_h=("emission_kg_h_canonical", "min"),
            median_emission_kg_h=("emission_kg_h_canonical", "median"),
            maximum_emission_kg_h=("emission_kg_h_canonical", "max"),
            maximum_assignment_distance_km=(
                "distance_to_site_centroid_km", "max"
            ),
        )
        .reset_index()
    )

    audit["ready_for_negative_search"] = (
        (audit["event_rows"] >= 2)
        & (audit["patch_rows"] >= 2)
        & (audit["unique_scenes"] >= 2)
    )

    audit_path = outputs / "513_selected_methaneair_positive_audit_v2.csv"
    audit.to_csv(audit_path, index=False)

    print("\nFIXED SITE CENTROIDS")
    print("=" * 110)
    print(
        selected_sites[
            [
                "candidate_site_id",
                "latitude",
                "longitude",
                "unique_events",
                "unique_dates",
                "downloaded_positive_patches",
            ]
        ].to_string(index=False)
    )

    print("\nPOSITIVE AUDIT V2")
    print("=" * 110)
    print(audit.to_string(index=False))

    print("\nMATCHED EVENTS")
    print("=" * 110)
    display_columns = [
        "site_id",
        "event_id_canonical",
        "latitude_canonical",
        "longitude_canonical",
        "distance_to_site_centroid_km",
        "datetime_utc_canonical",
        "emission_kg_h_canonical",
        "scene_id",
        "patch_exists_now",
    ]
    print(
        positive_manifest[display_columns]
        .sort_values(["site_id", "datetime_utc_canonical"])
        .to_string(index=False)
    )

    print("\nCreated:")
    for path in (
        selected_events_path,
        positive_manifest_path,
        audit_path,
        unmatched_path,
    ):
        print(path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
