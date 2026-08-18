from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import re

import numpy as np
import pandas as pd
from netCDF4 import Dataset, chartostring, num2date


PROJECT = Path("/Users/happydoraaa/methane_release_project")
NC_ROOT = PROJECT / "external_data/methaneair_controlled_release"
GT_PATH = PROJECT / "outputs/59_methaneair_ground_truth_with_source_coordinates.csv"

INVENTORY_PATH = PROJECT / "outputs/65_all_nc_inventory.csv"
CANDIDATES_PATH = PROJECT / "outputs/66_all45_nc_match_candidates.csv"
BEST_PATH = PROJECT / "outputs/67_all45_ground_truth_nc_mapping.csv"
SUMMARY_PATH = PROJECT / "outputs/68_all45_mapping_summary.txt"

FLIGHT_IDS = ("RF01E", "RF03E", "RF04", "RF05")

LAT_NAMES = {
    "latitude", "lat", "xlat", "nav_lat", "pixel_latitude",
}
LON_NAMES = {
    "longitude", "lon", "long", "xlon", "xlong",
    "nav_lon", "pixel_longitude",
}
TIME_NAMES = {
    "time", "times", "timestamp", "datetime",
    "time_utc", "utc_time", "acquisition_time",
}
SEGMENT_PATTERN = re.compile(
    r"(?i)(?:segment|seg|pass|lawn)[_\-\s]*0*(\d{1,3})(?!\d)"
)


def infer_flight_id(path: Path) -> str | None:
    text = path.as_posix().upper()
    for flight_id in FLIGHT_IDS:
        if flight_id in text:
            return flight_id
    return None


def walk_variables(group: Any, prefix: str = ""):
    for name, variable in group.variables.items():
        path = f"{prefix}/{name}" if prefix else f"/{name}"
        yield path, name, variable

    for group_name, subgroup in group.groups.items():
        next_prefix = f"{prefix}/{group_name}" if prefix else f"/{group_name}"
        yield from walk_variables(subgroup, next_prefix)


def coordinate_kind(name: str, variable: Any) -> str | None:
    lower = name.lower()
    units = str(getattr(variable, "units", "")).lower()
    standard_name = str(getattr(variable, "standard_name", "")).lower()
    long_name = str(getattr(variable, "long_name", "")).lower()

    if (
        lower in LAT_NAMES
        or standard_name == "latitude"
        or "degrees_north" in units
        or "latitude" in long_name
    ):
        return "latitude"

    if (
        lower in LON_NAMES
        or standard_name == "longitude"
        or "degrees_east" in units
        or "longitude" in long_name
    ):
        return "longitude"

    return None


def read_coordinate(variable: Any) -> np.ndarray:
    ndim = len(variable.shape)

    if ndim <= 2:
        raw = variable[:]
    else:
        # XLAT(Time, y, x) / XLONG(Time, y, x): read one time slice only.
        key = (0,) * (ndim - 2) + (slice(None), slice(None))
        raw = variable[key]

    values = np.ma.asarray(raw)

    if np.ma.isMaskedArray(values):
        values = values.filled(np.nan)

    return np.asarray(values, dtype=float).squeeze()


def normalize_source_longitude(
    source_lon: float,
    lon_values: np.ndarray,
) -> float:
    finite = lon_values[np.isfinite(lon_values)]

    if finite.size == 0:
        return source_lon

    if np.nanmin(finite) >= 0 and np.nanmax(finite) > 180:
        return source_lon % 360.0

    return source_lon


def haversine_m(
    latitude: np.ndarray,
    longitude: np.ndarray,
    source_latitude: float,
    source_longitude: float,
) -> np.ndarray:
    radius_m = 6_371_000.0

    lat1 = np.radians(latitude)
    lon1 = np.radians(longitude)
    lat2 = np.radians(source_latitude)
    lon2 = np.radians(source_longitude)

    dlat = lat1 - lat2
    dlon = lon1 - lon2

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    a = np.clip(a, 0.0, 1.0)
    return 2.0 * radius_m * np.arcsin(np.sqrt(a))


def evaluate_coordinate_pair(
    lat_path: str,
    lat_var: Any,
    lon_path: str,
    lon_var: Any,
    source_lat: float,
    source_lon: float,
) -> dict[str, Any] | None:
    try:
        lat = read_coordinate(lat_var)
        lon = read_coordinate(lon_var)
    except Exception:
        return None

    finite_lat = lat[np.isfinite(lat)]
    finite_lon = lon[np.isfinite(lon)]

    if finite_lat.size == 0 or finite_lon.size == 0:
        return None

    adjusted_source_lon = normalize_source_longitude(source_lon, lon)

    lat_min = float(np.nanmin(finite_lat))
    lat_max = float(np.nanmax(finite_lat))
    lon_min = float(np.nanmin(finite_lon))
    lon_max = float(np.nanmax(finite_lon))

    inside = (
        lat_min <= source_lat <= lat_max
        and lon_min <= adjusted_source_lon <= lon_max
    )

    nearest_row = pd.NA
    nearest_col = pd.NA
    nearest_lat = np.nan
    nearest_lon = np.nan
    nearest_distance = np.nan

    try:
        if lat.ndim == 1 and lon.ndim == 1:
            row = int(np.nanargmin(np.abs(lat - source_lat)))
            col = int(np.nanargmin(np.abs(lon - adjusted_source_lon)))
            nearest_row = row
            nearest_col = col
            nearest_lat = float(lat[row])
            nearest_lon = float(lon[col])
            nearest_distance = float(
                haversine_m(
                    np.asarray(nearest_lat),
                    np.asarray(nearest_lon),
                    source_lat,
                    adjusted_source_lon,
                )
            )

        else:
            lat_grid = None
            lon_grid = None

            if lat.shape == lon.shape:
                lat_grid = lat
                lon_grid = lon

            elif lat.ndim == 1 and lon.ndim == 2 and lon.shape[0] == lat.size:
                lat_grid = np.broadcast_to(lat[:, None], lon.shape)
                lon_grid = lon

            elif lat.ndim == 2 and lon.ndim == 1 and lat.shape[1] == lon.size:
                lat_grid = lat
                lon_grid = np.broadcast_to(lon[None, :], lat.shape)

            if lat_grid is not None and lon_grid is not None:
                distance = haversine_m(
                    lat_grid,
                    lon_grid,
                    source_lat,
                    adjusted_source_lon,
                )
                flat_index = int(np.nanargmin(distance))
                row, col = np.unravel_index(flat_index, distance.shape)

                nearest_row = int(row)
                nearest_col = int(col)
                nearest_lat = float(lat_grid[row, col])
                nearest_lon = float(lon_grid[row, col])
                nearest_distance = float(distance[row, col])

    except Exception:
        pass

    return {
        "latitude_variable_path": lat_path,
        "longitude_variable_path": lon_path,
        "latitude_min": lat_min,
        "latitude_max": lat_max,
        "longitude_min": lon_min,
        "longitude_max": lon_max,
        "source_inside_scene": bool(inside),
        "nearest_grid_row": nearest_row,
        "nearest_grid_column": nearest_col,
        "nearest_grid_latitude": nearest_lat,
        "nearest_grid_longitude": nearest_lon,
        "distance_source_to_grid_pixel_m": nearest_distance,
    }


def parse_string_times(values: Any) -> list[pd.Timestamp]:
    array = np.asarray(values)

    try:
        if array.dtype.kind in {"S", "U"} and array.ndim >= 2:
            strings = chartostring(array)
        else:
            strings = array
    except Exception:
        strings = array

    output: list[pd.Timestamp] = []

    for value in np.ravel(strings):
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")

        text = str(value).strip().replace("_", " ")

        if not text:
            continue

        timestamp = pd.to_datetime(text, utc=True, errors="coerce")

        if not pd.isna(timestamp):
            output.append(pd.Timestamp(timestamp))

    return output


def parse_numeric_times(variable: Any) -> list[pd.Timestamp]:
    units = str(getattr(variable, "units", ""))
    calendar = str(getattr(variable, "calendar", "standard"))

    if "since" not in units.lower():
        return []

    raw = np.ma.asarray(variable[:])

    if np.ma.isMaskedArray(raw):
        raw = raw.compressed()
    else:
        raw = np.ravel(raw)

    if raw.size == 0:
        return []

    try:
        decoded = num2date(
            raw,
            units=units,
            calendar=calendar,
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=False,
        )
    except Exception:
        return []

    output: list[pd.Timestamp] = []

    for value in np.ravel(decoded):
        try:
            timestamp = pd.Timestamp(
                year=value.year,
                month=value.month,
                day=value.day,
                hour=getattr(value, "hour", 0),
                minute=getattr(value, "minute", 0),
                second=getattr(value, "second", 0),
                microsecond=getattr(value, "microsecond", 0),
                tz="UTC",
            )
            output.append(timestamp)
        except Exception:
            timestamp = pd.to_datetime(str(value), utc=True, errors="coerce")
            if not pd.isna(timestamp):
                output.append(pd.Timestamp(timestamp))

    return output


def extract_times(dataset: Any) -> tuple[list[pd.Timestamp], list[str]]:
    times: list[pd.Timestamp] = []
    sources: list[str] = []

    for variable_path, name, variable in walk_variables(dataset):
        lower = name.lower()
        units = str(getattr(variable, "units", "")).lower()

        parsed: list[pd.Timestamp] = []

        if lower in TIME_NAMES or "timestamp" in lower or lower.endswith("_time"):
            if "since" in units:
                parsed = parse_numeric_times(variable)
            else:
                try:
                    parsed = parse_string_times(variable[:])
                except Exception:
                    parsed = []

        elif "since" in units and "time" in lower:
            parsed = parse_numeric_times(variable)

        if parsed:
            times.extend(parsed)
            sources.append(variable_path)

    for attr_name in (
        "time_coverage_start",
        "time_coverage_end",
        "start_time",
        "end_time",
        "acquisition_time",
        "acquisition_datetime",
    ):
        if attr_name not in dataset.ncattrs():
            continue

        value = dataset.getncattr(attr_name)
        timestamp = pd.to_datetime(
            str(value).replace("_", " "),
            utc=True,
            errors="coerce",
        )

        if not pd.isna(timestamp):
            times.append(pd.Timestamp(timestamp))
            sources.append(f"@{attr_name}")

    return sorted(set(times)), sorted(set(sources))


def extract_segment_candidates(path: Path, dataset: Any) -> list[int]:
    candidates = {
        int(match)
        for match in SEGMENT_PATTERN.findall(path.as_posix())
    }

    for attr_name in dataset.ncattrs():
        lower = attr_name.lower()

        if lower not in {
            "segment", "seg", "pass",
            "segment_id", "segment_number",
        }:
            continue

        try:
            candidates.add(int(float(dataset.getncattr(attr_name))))
        except Exception:
            pass

    for _, name, variable in walk_variables(dataset):
        lower = name.lower()

        if lower not in {
            "segment", "seg", "pass",
            "segment_id", "segment_number",
        }:
            continue

        try:
            values = np.asarray(variable[:]).squeeze()
            if values.size == 1:
                candidates.add(int(float(values)))
        except Exception:
            pass

    return sorted(candidates)


def classify_file_role(
    path: Path,
    variable_names: set[str],
    segments: list[int],
) -> str:
    path_text = path.as_posix().lower()

    if {"xlat", "xlong", "u", "v", "ph", "phb"}.issubset(variable_names):
        return "wrf_auxiliary"

    if "segment_raster" in path_text or "segment_rasters" in path_text:
        return "segment_raster"

    if segments:
        return "segment_product"

    if any(
        token in name
        for name in variable_names
        for token in ("xch4", "methane", "ch4", "enhancement", "plume")
    ):
        return "methane_product"

    return "generic_netcdf"


def choose_best_coordinate_pair(
    latitude_candidates: list[tuple[str, Any]],
    longitude_candidates: list[tuple[str, Any]],
    source_lat: float,
    source_lon: float,
) -> dict[str, Any]:
    evaluations: list[dict[str, Any]] = []

    for lat_path, lat_var in latitude_candidates:
        for lon_path, lon_var in longitude_candidates:
            result = evaluate_coordinate_pair(
                lat_path,
                lat_var,
                lon_path,
                lon_var,
                source_lat,
                source_lon,
            )
            if result is not None:
                evaluations.append(result)

    if not evaluations:
        return {
            "coordinate_status": "missing_or_unreadable_coordinates",
            "source_inside_scene": pd.NA,
        }

    def key(item: dict[str, Any]):
        inside_rank = 0 if item["source_inside_scene"] else 1
        distance = item["distance_source_to_grid_pixel_m"]
        distance_rank = (
            float(distance)
            if pd.notna(distance)
            else float("inf")
        )
        return inside_rank, distance_rank

    best = min(evaluations, key=key)
    best["coordinate_status"] = (
        "inside" if best["source_inside_scene"] else "outside"
    )
    return best


def nearest_time(
    gt_time: pd.Timestamp,
    file_times: list[pd.Timestamp],
) -> tuple[pd.Timestamp | pd.NaT, float]:
    if not file_times:
        return pd.NaT, float("nan")

    differences = np.asarray(
        [abs((value - gt_time).total_seconds()) for value in file_times],
        dtype=float,
    )
    index = int(np.argmin(differences))
    return file_times[index], float(differences[index])


def mapping_status(row: pd.Series) -> str:
    inside_value = row.get("source_inside_scene")
    segment_value = row.get("segment_match")
    diff = row.get("absolute_time_difference_seconds")

    inside_known = pd.notna(inside_value)
    segment_known = pd.notna(segment_value)

    inside = bool(inside_value) if inside_known else None
    segment_match = bool(segment_value) if segment_known else None

    if inside is False:
        return "review_source_outside_scene"

    if pd.notna(diff) and float(diff) <= 60:
        if segment_match is True:
            return "matched_segment_and_time_60s"
        return "matched_time_60s"

    if pd.notna(diff) and float(diff) <= 600:
        if segment_match is True:
            return "matched_segment_and_time_10min"
        return "matched_time_10min"

    if segment_match is True and inside is True:
        return "matched_segment_only"

    return "needs_manual_review"


if not GT_PATH.exists():
    raise FileNotFoundError(f"Ground-truth file not found: {GT_PATH}")

ground_truth = pd.read_csv(GT_PATH)
ground_truth["timestamp_utc"] = pd.to_datetime(
    ground_truth["timestamp_utc"],
    utc=True,
    errors="raise",
)

required_columns = {
    "record_id",
    "flight_id",
    "segment",
    "timestamp_utc",
    "source_latitude",
    "source_longitude",
    "metered_release_rate_kg_hr",
    "physical_release_gt",
}
missing_columns = required_columns - set(ground_truth.columns)

if missing_columns:
    raise ValueError(
        f"Missing ground-truth columns: {sorted(missing_columns)}"
    )

source_by_flight = (
    ground_truth.groupby("flight_id")[
        ["source_latitude", "source_longitude"]
    ]
    .first()
    .to_dict("index")
)

nc_files = sorted(NC_ROOT.rglob("*.nc"))

print(f"Scanning {len(nc_files)} NetCDF files...")

inventory_records: list[dict[str, Any]] = []
times_by_file: dict[str, list[pd.Timestamp]] = {}

for number, path in enumerate(nc_files, start=1):
    print(f"[{number}/{len(nc_files)}] {path}")

    flight_id = infer_flight_id(path)

    record: dict[str, Any] = {
        "nc_path": str(path),
        "flight_id": flight_id,
        "file_size_bytes": path.stat().st_size,
    }

    if flight_id not in source_by_flight:
        record["scan_status"] = "unknown_flight_id"
        inventory_records.append(record)
        times_by_file[str(path)] = []
        continue

    source_lat = float(source_by_flight[flight_id]["source_latitude"])
    source_lon = float(source_by_flight[flight_id]["source_longitude"])

    try:
        with Dataset(path) as dataset:
            variables = list(walk_variables(dataset))
            variable_names = {name.lower() for _, name, _ in variables}

            latitude_candidates = []
            longitude_candidates = []

            for variable_path, name, variable in variables:
                kind = coordinate_kind(name, variable)

                if kind == "latitude":
                    latitude_candidates.append((variable_path, variable))
                elif kind == "longitude":
                    longitude_candidates.append((variable_path, variable))

            coordinate_result = choose_best_coordinate_pair(
                latitude_candidates,
                longitude_candidates,
                source_lat,
                source_lon,
            )

            file_times, time_sources = extract_times(dataset)
            segment_candidates = extract_segment_candidates(path, dataset)
            file_role = classify_file_role(
                path,
                variable_names,
                segment_candidates,
            )

            record.update(coordinate_result)
            record.update({
                "scan_status": "ok",
                "file_role": file_role,
                "segment_candidates": json.dumps(segment_candidates),
                "time_count": len(file_times),
                "time_start_utc": (
                    file_times[0].isoformat() if file_times else ""
                ),
                "time_end_utc": (
                    file_times[-1].isoformat() if file_times else ""
                ),
                "time_variable_sources": " | ".join(time_sources),
                "latitude_candidate_count": len(latitude_candidates),
                "longitude_candidate_count": len(longitude_candidates),
            })

            times_by_file[str(path)] = file_times

    except Exception as exc:
        record["scan_status"] = f"error:{type(exc).__name__}"
        record["error_message"] = str(exc)
        times_by_file[str(path)] = []

    inventory_records.append(record)

inventory = pd.DataFrame(inventory_records)
INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
inventory.to_csv(INVENTORY_PATH, index=False)

role_rank = {
    "segment_raster": 0,
    "segment_product": 1,
    "methane_product": 2,
    "generic_netcdf": 3,
    "wrf_auxiliary": 4,
}

candidate_records: list[dict[str, Any]] = []

for _, gt in ground_truth.iterrows():
    same_flight = inventory[
        (inventory["flight_id"] == gt["flight_id"])
        & (inventory["scan_status"] == "ok")
    ]

    for _, nc in same_flight.iterrows():
        file_times = times_by_file.get(str(nc["nc_path"]), [])
        matched_time, time_difference = nearest_time(
            gt["timestamp_utc"],
            file_times,
        )

        try:
            segments = json.loads(nc.get("segment_candidates", "[]"))
        except Exception:
            segments = []

        gt_segment = int(gt["segment"])

        if segments:
            segment_match = gt_segment in {
                int(value) for value in segments
            }
        else:
            segment_match = None

        source_inside = nc.get("source_inside_scene")

        if pd.isna(source_inside):
            inside_rank = 1
        elif bool(source_inside):
            inside_rank = 0
        else:
            inside_rank = 2

        if segment_match is True:
            segment_rank = 0
        elif segment_match is None:
            segment_rank = 1
        else:
            segment_rank = 2

        time_rank = (
            float(time_difference)
            if pd.notna(time_difference)
            else float("inf")
        )

        candidate_records.append({
            "record_id": gt["record_id"],
            "flight_id": gt["flight_id"],
            "segment": gt_segment,
            "ground_truth_timestamp_utc":
                gt["timestamp_utc"].isoformat(),
            "physical_release_gt":
                int(gt["physical_release_gt"]),
            "metered_release_rate_kg_hr":
                gt["metered_release_rate_kg_hr"],
            "source_latitude": gt["source_latitude"],
            "source_longitude": gt["source_longitude"],
            "nc_path": nc["nc_path"],
            "file_role": nc.get("file_role"),
            "segment_candidates": nc.get("segment_candidates"),
            "segment_match": segment_match,
            "nearest_nc_time_utc": (
                matched_time.isoformat()
                if pd.notna(matched_time)
                else ""
            ),
            "absolute_time_difference_seconds": time_difference,
            "source_inside_scene": source_inside,
            "latitude_variable_path":
                nc.get("latitude_variable_path"),
            "longitude_variable_path":
                nc.get("longitude_variable_path"),
            "nearest_grid_row":
                nc.get("nearest_grid_row"),
            "nearest_grid_column":
                nc.get("nearest_grid_column"),
            "nearest_grid_latitude":
                nc.get("nearest_grid_latitude"),
            "nearest_grid_longitude":
                nc.get("nearest_grid_longitude"),
            "distance_source_to_grid_pixel_m":
                nc.get("distance_source_to_grid_pixel_m"),
            "_segment_rank": segment_rank,
            "_role_rank": role_rank.get(
                str(nc.get("file_role")),
                99,
            ),
            "_inside_rank": inside_rank,
            "_time_rank": time_rank,
        })

candidates = pd.DataFrame(candidate_records)

if candidates.empty:
    raise RuntimeError(
        "No same-flight NetCDF candidates were found."
    )

candidates = candidates.sort_values(
    [
        "record_id",
        "_segment_rank",
        "_role_rank",
        "_inside_rank",
        "_time_rank",
        "nc_path",
    ],
    kind="stable",
)

candidates.to_csv(CANDIDATES_PATH, index=False)

best = (
    candidates
    .drop_duplicates(subset=["record_id"], keep="first")
    .copy()
)

best["mapping_status"] = best.apply(mapping_status, axis=1)

best = ground_truth.merge(
    best.drop(
        columns=[
            "flight_id",
            "segment",
            "physical_release_gt",
            "metered_release_rate_kg_hr",
            "source_latitude",
            "source_longitude",
            "_segment_rank",
            "_role_rank",
            "_inside_rank",
            "_time_rank",
        ],
        errors="ignore",
    ),
    on="record_id",
    how="left",
    validate="one_to_one",
)

best.to_csv(BEST_PATH, index=False)

summary_lines = [
    "MethaneAIR 45-row ground-truth to NetCDF mapping summary",
    "=" * 70,
    f"Ground-truth rows: {len(ground_truth)}",
    f"Mapped output rows: {len(best)}",
    f"NetCDF files scanned: {len(inventory)}",
    "",
    "Mapping status:",
    best["mapping_status"]
    .value_counts(dropna=False)
    .to_string(),
    "",
    "By flight and mapping status:",
    best.groupby(
        ["flight_id", "mapping_status"],
        dropna=False,
    )
    .size()
    .reset_index(name="rows")
    .to_string(index=False),
    "",
    "Time difference summary (seconds):",
    best["absolute_time_difference_seconds"]
    .describe()
    .to_string(),
    "",
    "Rows needing manual review:",
]

review = best[
    best["mapping_status"].isin(
        [
            "needs_manual_review",
            "review_source_outside_scene",
        ]
    )
]

if review.empty:
    summary_lines.append("NONE")
else:
    summary_lines.append(
        review[
            [
                "record_id",
                "flight_id",
                "segment",
                "mapping_status",
                "absolute_time_difference_seconds",
                "nc_path",
            ]
        ].to_string(index=False)
    )

SUMMARY_PATH.write_text(
    "\n".join(summary_lines),
    encoding="utf-8",
)

print("\nCreated:")
print(INVENTORY_PATH)
print(CANDIDATES_PATH)
print(BEST_PATH)
print(SUMMARY_PATH)

print("\nMapping status:")
print(
    best["mapping_status"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nBy flight:")
print(
    best.groupby(
        ["flight_id", "mapping_status"],
        dropna=False,
    )
    .size()
    .reset_index(name="rows")
    .to_string(index=False)
)

print("\nRows needing manual review:")
if review.empty:
    print("NONE")
else:
    print(
        review[
            [
                "record_id",
                "flight_id",
                "segment",
                "mapping_status",
                "absolute_time_difference_seconds",
                "nc_path",
            ]
        ].to_string(index=False)
    )
