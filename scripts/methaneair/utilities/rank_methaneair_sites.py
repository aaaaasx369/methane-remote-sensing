from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

ROOT = Path("/Users/happydoraaa/methane_release_project")
AVAILABILITY = ROOT / "outputs/15_methaneair_s2_landsat_availability.csv"
PATCH_TABLE = ROOT / "outputs/18_methaneair_s2_dataset_table.csv"
OUTPUT = ROOT / "outputs/510_methaneair_candidate_sites.csv"

# 1 km approximately expressed in radians on Earth.
EARTH_RADIUS_KM = 6371.0088
CLUSTER_RADIUS_KM = 1.0

events = pd.read_csv(AVAILABILITY)

required = ["event_id", "lat", "lon", "datetime_utc", "emission_kg_hr"]
missing = [c for c in required if c not in events.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

events["lat"] = pd.to_numeric(events["lat"], errors="coerce")
events["lon"] = pd.to_numeric(events["lon"], errors="coerce")
events["emission_kg_hr"] = pd.to_numeric(
    events["emission_kg_hr"], errors="coerce"
)
events["datetime_utc"] = pd.to_datetime(
    events["datetime_utc"], errors="coerce", utc=True
)

events = events.dropna(
    subset=["lat", "lon", "datetime_utc", "emission_kg_hr"]
).copy()

# Keep events for which the earlier GEE availability search found S2.
s2_count_columns = [
    c for c in [
        "s2_count_pm1day",
        "s2_count",
    ]
    if c in events.columns
]

if s2_count_columns:
    s2_available = np.zeros(len(events), dtype=bool)
    for column in s2_count_columns:
        values = pd.to_numeric(events[column], errors="coerce").fillna(0)
        s2_available |= values.to_numpy() > 0
    events = events[s2_available].copy()

# DBSCAN with haversine distance.
coordinates_rad = np.radians(events[["lat", "lon"]].to_numpy())

clustering = DBSCAN(
    eps=CLUSTER_RADIUS_KM / EARTH_RADIUS_KM,
    min_samples=1,
    metric="haversine",
).fit(coordinates_rad)

events["site_cluster"] = clustering.labels_

# Check which events already have a downloaded positive patch.
downloaded_event_ids = set()

if PATCH_TABLE.exists():
    patches = pd.read_csv(PATCH_TABLE)

    if "event_id" in patches.columns:
        downloaded_event_ids = set(
            patches["event_id"].dropna().astype(str)
        )

events["has_downloaded_patch"] = (
    events["event_id"].astype(str).isin(downloaded_event_ids)
)

summary = (
    events.groupby("site_cluster")
    .agg(
        latitude=("lat", "median"),
        longitude=("lon", "median"),
        event_rows=("event_id", "size"),
        unique_events=("event_id", "nunique"),
        unique_dates=("datetime_utc", lambda x: x.dt.date.nunique()),
        downloaded_positive_patches=("has_downloaded_patch", "sum"),
        minimum_emission_kg_hr=("emission_kg_hr", "min"),
        median_emission_kg_hr=("emission_kg_hr", "median"),
        maximum_emission_kg_hr=("emission_kg_hr", "max"),
        first_event_time=("datetime_utc", "min"),
        last_event_time=("datetime_utc", "max"),
    )
    .reset_index()
)

summary["candidate_site_id"] = summary["site_cluster"].map(
    lambda x: f"MethaneAIR_site_{x:03d}"
)

summary["eligible_initially"] = (
    (summary["unique_events"] >= 2)
    & (summary["downloaded_positive_patches"] >= 2)
)

summary = summary.sort_values(
    [
        "eligible_initially",
        "downloaded_positive_patches",
        "unique_events",
        "median_emission_kg_hr",
    ],
    ascending=[False, False, False, False],
)

summary.to_csv(OUTPUT, index=False)

print(summary.head(30).to_string(index=False))
print(f"\nSaved: {OUTPUT}")
