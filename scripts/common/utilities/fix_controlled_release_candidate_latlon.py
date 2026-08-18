from pathlib import Path
import json
import re
import numpy as np
import pandas as pd


CANDIDATE_PATH = Path("outputs/12_dataset_candidate_events.csv")
EVENT_PATH = Path("outputs/10_final_events_for_gee.csv")
OUT_PATH = Path("outputs/12_dataset_candidate_events_with_latlon.csv")


def parse_geo_point(value):
    """
    Parse GEE .geo column.
    Usually looks like:
    {"type":"Point","coordinates":[lon, lat]}
    """
    if pd.isna(value):
        return np.nan, np.nan

    text = str(value)

    # Method 1: JSON parse
    try:
        geo = json.loads(text)
        if geo.get("type") == "Point":
            lon, lat = geo.get("coordinates", [np.nan, np.nan])[:2]
            return float(lat), float(lon)
    except Exception:
        pass

    # Method 2: regex fallback
    try:
        nums = re.findall(r"-?\d+\.\d+|-?\d+", text)
        nums = [float(x) for x in nums]

        # For Point geometry, usually the first two are lon, lat
        if len(nums) >= 2:
            lon, lat = nums[0], nums[1]
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
    except Exception:
        pass

    return np.nan, np.nan


def main():
    if not CANDIDATE_PATH.exists():
        raise FileNotFoundError(f"Cannot find {CANDIDATE_PATH}")

    cand = pd.read_csv(CANDIDATE_PATH)
    print("Candidate shape:", cand.shape)
    print("Candidate columns:", cand.columns.tolist())

    # Step A: parse lat/lon from .geo if possible
    if ".geo" in cand.columns:
        parsed = cand[".geo"].apply(parse_geo_point)
        cand["lat_from_geo"] = parsed.apply(lambda x: x[0])
        cand["lon_from_geo"] = parsed.apply(lambda x: x[1])
    else:
        cand["lat_from_geo"] = np.nan
        cand["lon_from_geo"] = np.nan

    cand["lat"] = cand["lat_from_geo"]
    cand["lon"] = cand["lon_from_geo"]

    # Step B: fallback merge with 10_final_events_for_gee.csv
    if EVENT_PATH.exists():
        events = pd.read_csv(EVENT_PATH)

        keep_cols = [
            c for c in [
                "event_id",
                "datetime_utc",
                "paper",
                "satellite_from_paper",
                "lat",
                "lon",
                "site_name"
            ]
            if c in events.columns
        ]

        events_small = events[keep_cols].copy()
        events_small = events_small.rename(columns={
            "lat": "lat_from_event_table",
            "lon": "lon_from_event_table",
            "site_name": "site_name_from_event_table"
        })

        merge_keys = [c for c in ["event_id", "datetime_utc", "paper", "satellite_from_paper"]
                      if c in cand.columns and c in events_small.columns]

        print("Merge keys:", merge_keys)

        if merge_keys:
            cand = cand.merge(
                events_small,
                on=merge_keys,
                how="left"
            )

            cand["lat"] = cand["lat"].fillna(cand["lat_from_event_table"])
            cand["lon"] = cand["lon"].fillna(cand["lon_from_event_table"])

            if "site_name_from_event_table" in cand.columns:
                cand["site_name"] = cand["site_name_from_event_table"]

    # Clean numeric columns
    cand["lat"] = pd.to_numeric(cand["lat"], errors="coerce")
    cand["lon"] = pd.to_numeric(cand["lon"], errors="coerce")

    print("\nValid lat count:", cand["lat"].notna().sum(), "/", len(cand))
    print("Valid lon count:", cand["lon"].notna().sum(), "/", len(cand))

    if "label" in cand.columns:
        print("\nLabel counts:")
        print(cand["label"].value_counts(dropna=False))

    if "s2_count" in cand.columns:
        print("\nS2 count > 0:", (pd.to_numeric(cand["s2_count"], errors="coerce").fillna(0) > 0).sum())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cand.to_csv(OUT_PATH, index=False)

    print("\nSaved:", OUT_PATH)
    print("Output shape:", cand.shape)


if __name__ == "__main__":
    main()