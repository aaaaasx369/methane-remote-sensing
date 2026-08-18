#!/usr/bin/env python3
"""
Start methane dataset expansion from the current unified master.

Outputs
-------
012_unified_methane_master_fixed.csv
013_unified_methane_counts_fixed.csv
014_dataset_recovery_candidates.csv
015_dataset_recovery_summary.csv
016_methaneair_observation_inventory.csv            (when a candidate is found)
017_methaneair_s2_search_requests.csv                (when inventory is available)
018_methaneair_temporal_negative_requests.csv        (when inventory is available)
019_carbonmapper_observation_inventory.csv           (when a candidate is found)
020_carbonmapper_satellite_search_requests.csv       (when inventory is available)
021_historical_multisatellite_inventory.csv          (when a candidate is found)

This script does not label temporal candidates as confirmed negatives. It creates
requests that must later pass release/plume/QA exclusion checks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


TARGETS = {
    "methaneair_baseline_s2_110": {
        "expected_rows": [110],
        "expected_names": [
            "18_methaneair_s2_dataset_table.csv",
            "methaneair_s2_dataset_table.csv",
        ],
        "keywords": [
            "methaneair", "sentinel", "s2", "plume", "label",
            "acquisition", "latitude", "longitude", "image", "path",
        ],
    },
    "controlled_release_s2_76": {
        "expected_rows": [76],
        "expected_names": [
            "20_controlled_release_s2_patch_index.csv",
            "controlled_release_s2_patch_index.csv",
        ],
        "keywords": [
            "controlled", "release", "sentinel", "s2", "label",
            "acquisition", "latitude", "longitude", "image", "path",
        ],
    },
    "methaneair_observations_435": {
        "expected_rows": [435],
        "expected_names": [
            "methaneair_435.csv",
            "methaneair_inventory.csv",
            "methaneair_catalog.csv",
        ],
        "keywords": [
            "methaneair", "plume", "flight", "date", "time",
            "latitude", "longitude", "emission", "location",
        ],
    },
    "carbonmapper_observations_226": {
        "expected_rows": [226, 193],
        "expected_names": [
            "carbon_mapper_observations.csv",
            "carbon_mapper_observation_level.csv",
            "carbon_mapper_classification.csv",
        ],
        "keywords": [
            "carbon", "mapper", "classification", "tp", "fn", "tn", "fp",
            "date", "time", "latitude", "longitude", "emission",
        ],
    },
    "historical_multisatellite_17": {
        "expected_rows": [17, 44, 88],
        "expected_names": [
            "historical_multisatellite.csv",
            "multisatellite_acquisitions.csv",
        ],
        "keywords": [
            "satellite", "acquisition", "team", "classification",
            "ground", "truth", "emission", "date",
        ],
    },
}


ALIASES = {
    "record_id": [
        "record_id", "sample_id", "observation_id", "event_id", "plume_id",
        "source_row_id", "id",
    ],
    "site_id": [
        "site_id", "site", "site_name", "location_id", "location",
        "facility_site", "master_site",
    ],
    "facility_id": [
        "facility_id", "facility", "facility_name", "source_id", "asset_id",
    ],
    "latitude": [
        "latitude", "lat", "source_latitude", "plume_latitude",
        "release_latitude", "center_lat",
    ],
    "longitude": [
        "longitude", "lon", "lng", "source_longitude", "plume_longitude",
        "release_longitude", "center_lon",
    ],
    "acquisition_time_utc": [
        "acquisition_time_utc", "acquisition_time", "scene_timestamp",
        "timestamp", "datetime", "date_time", "overpass_time",
        "flight_time", "date",
    ],
    "label": [
        "label", "ground_truth", "target", "binary_label",
        "classification", "tc_classification", "result", "status",
    ],
    "emission_rate_kg_hr": [
        "emission_rate_kg_hr", "emission_kg_hr", "release_rate_kg_hr",
        "ground_truth_rate_kg_hr", "emission_auto", "emission_rate",
        "release_rate",
    ],
    "sensor": [
        "sensor", "satellite", "instrument", "platform",
    ],
    "scene_id": [
        "scene_id", "image_id", "product_id", "granule_id",
        "landsat_scene_id", "s2_scene_id",
    ],
    "image_path": [
        "image_path", "patch_path", "tif_path", "local_path",
        "raster_path", "file_path", "s2_0_path", "l89_0_path",
    ],
    "wind_speed_m_s": [
        "wind_speed_m_s", "wind_speed", "wind_speed_avg_auto",
    ],
    "wind_direction_deg": [
        "wind_direction_deg", "wind_direction", "wind_direction_avg_auto",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/project/6002520/yunjung1/MethaneFuse"),
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        default=Path("/project/6002520/yunjung1"),
    )
    parser.add_argument(
        "--master",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--max-csv-size-mb",
        type=float,
        default=250.0,
    )
    return parser.parse_args()


def norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def normalized_columns(columns: Iterable[Any]) -> list[str]:
    return [norm(column) for column in columns]


def first_column(df: pd.DataFrame, aliases: Iterable[str]) -> pd.Series:
    mapping = {norm(column): column for column in df.columns}
    for alias in aliases:
        if norm(alias) in mapping:
            return df[mapping[norm(alias)]]
    return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")


def clean_text(series: pd.Series) -> pd.Series:
    output = series.astype("string").str.strip()
    return output.mask(
        output.isin(["", "nan", "None", "<NA>", "NaT"]),
        pd.NA,
    )


def normalize_label(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        try:
            number = int(value)
        except Exception:
            return pd.NA
        return number if number in (0, 1) else pd.NA

    text = str(value).strip().lower().replace("_", " ")
    positives = {
        "1", "positive", "pos", "plume", "detected", "tp",
        "true positive", "release on", "on",
    }
    negatives = {
        "0", "negative", "neg", "no plume", "not detected", "tn",
        "true negative", "release off", "off",
    }
    if text in positives:
        return 1
    if text in negatives:
        return 0

    # Provider classifications can be converted where unambiguous.
    if re.fullmatch(r"tp|true positive", text):
        return 1
    if re.fullmatch(r"fn|false negative", text):
        return 1
    if re.fullmatch(r"tn|true negative", text):
        return 0
    if re.fullmatch(r"fp|false positive", text):
        return 0

    return pd.NA


DATE_PATTERNS = [
    re.compile(r"LC0[89]_L2S[A-Z]_\d{6}_(20\d{6})_", re.I),
    re.compile(r"LT0[89]_L2S[A-Z]_\d{6}_(20\d{6})_", re.I),
    re.compile(r"(?<!\d)(20\d{2})[-_](\d{2})[-_](\d{2})(?!\d)"),
    re.compile(r"(?<!\d)(20\d{6})(?!\d)"),
]


def parse_date_from_text(value: Any) -> pd.Timestamp | None:
    if pd.isna(value):
        return None
    text = str(value)

    for pattern in DATE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue

        groups = match.groups()
        if len(groups) == 1:
            raw = groups[0]
        else:
            raw = "".join(groups)

        parsed = pd.to_datetime(raw, format="%Y%m%d", errors="coerce", utc=True)
        if pd.notna(parsed):
            return parsed

    # Last resort for ISO dates embedded in text.
    match = re.search(r"(20\d{2}-\d{2}-\d{2})", text)
    if match:
        parsed = pd.to_datetime(match.group(1), errors="coerce", utc=True)
        if pd.notna(parsed):
            return parsed

    return None


def raster_date_from_tags(path_value: Any) -> pd.Timestamp | None:
    if pd.isna(path_value):
        return None
    path = Path(str(path_value))
    if not path.exists():
        return None

    try:
        import rasterio
    except ImportError:
        return None

    try:
        with rasterio.open(path) as src:
            tags = src.tags()
            candidate_keys = [
                "ACQUISITION_DATE", "DATE_ACQUIRED", "SENSING_TIME",
                "DATATAKE_SENSING_START", "TIFFTAG_DATETIME",
                "acquisition_time", "scene_timestamp",
            ]
            for key in candidate_keys:
                if key in tags:
                    parsed = pd.to_datetime(tags[key], errors="coerce", utc=True)
                    if pd.notna(parsed):
                        return parsed

            for value in tags.values():
                parsed = parse_date_from_text(value)
                if parsed is not None:
                    return parsed
    except Exception:
        return None

    return None


def fix_landsat_master(master: pd.DataFrame) -> pd.DataFrame:
    fixed = master.copy()

    if "model_family" not in fixed.columns:
        return fixed

    mask = fixed["model_family"].astype("string").eq("landsat_temporal")
    if not mask.any():
        return fixed

    candidate_columns = [
        column for column in [
            "scene_id", "source_row_ids", "image_path",
            "t0_path", "t90_path", "t360_path",
        ]
        if column in fixed.columns
    ]

    for index in fixed.index[mask]:
        current = pd.to_datetime(
            fixed.at[index, "acquisition_time_utc"]
            if "acquisition_time_utc" in fixed.columns
            else pd.NA,
            errors="coerce",
            utc=True,
        )

        parsed: pd.Timestamp | None = None

        if pd.isna(current):
            for column in candidate_columns:
                parsed = parse_date_from_text(fixed.at[index, column])
                if parsed is not None:
                    break

        if parsed is None and "t0_path" in fixed.columns:
            parsed = raster_date_from_tags(fixed.at[index, "t0_path"])

        if parsed is not None:
            fixed.at[index, "acquisition_time_utc"] = (
                parsed.strftime("%Y-%m-%dT%H:%M:%SZ")
            )

        site = str(fixed.at[index, "site_id"]) if "site_id" in fixed.columns else "unknown_site"
        source_id = (
            str(fixed.at[index, "source_row_ids"])
            if "source_row_ids" in fixed.columns
            else str(fixed.at[index, "master_id"])
        )

        date_value = pd.to_datetime(
            fixed.at[index, "acquisition_time_utc"],
            errors="coerce",
            utc=True,
        )

        if pd.notna(date_value):
            event_id = f"{site}|{date_value.strftime('%Y-%m-%d')}"
        else:
            event_id = f"{site}|{source_id}"

        fixed.at[index, "event_group_id"] = event_id

        # Keep a useful Landsat product as scene_id where available.
        if (
            "scene_id" in fixed.columns
            and pd.isna(fixed.at[index, "scene_id"])
        ):
            for column in candidate_columns:
                value = fixed.at[index, column]
                if pd.isna(value):
                    continue
                match = re.search(
                    r"(L[CEOT]0[89]_L2S[A-Z]_\d{6}_20\d{6}_20\d{6}_\d{2}_T\d)",
                    str(value),
                    flags=re.I,
                )
                if match:
                    fixed.at[index, "scene_id"] = match.group(1)
                    break

    return fixed


def write_fixed_counts(master: pd.DataFrame, output: Path) -> None:
    records = []

    def add(group_type: str, group_value: str, subset: pd.DataFrame) -> None:
        labels = pd.to_numeric(subset["label"], errors="coerce")
        records.append({
            "group_type": group_type,
            "group_value": group_value,
            "rows": len(subset),
            "unique_events": subset["event_group_id"].nunique(dropna=True),
            "model_ready": int(
                subset.get("model_ready", False)
                .astype("string")
                .str.lower()
                .eq("true")
                .sum()
            ) if "model_ready" in subset.columns else 0,
            "qa_pass": int(
                subset.get("qa_pass", False)
                .astype("string")
                .str.lower()
                .eq("true")
                .sum()
            ) if "qa_pass" in subset.columns else 0,
            "positive": int((labels == 1).sum()),
            "negative": int((labels == 0).sum()),
            "unlabeled": int(labels.isna().sum()),
        })

    add("all", "all", master)

    for column in ["sensor", "model_family", "ground_truth_source"]:
        if column not in master.columns:
            continue
        for value, subset in master.groupby(
            master[column].fillna(f"missing_{column}"),
            dropna=False,
        ):
            add(column, str(value), subset)

    pd.DataFrame(records).to_csv(output, index=False)


def csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def file_header(path: Path) -> list[str]:
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.reader(handle)
            return next(reader, [])


def score_candidate(
    target_name: str,
    path: Path,
    rows: int,
    columns: list[str],
) -> tuple[float, list[str]]:
    spec = TARGETS[target_name]
    path_text = norm(str(path))
    column_tokens = set(normalized_columns(columns))

    score = 0.0
    reasons = []

    if path.name.lower() in {
        name.lower() for name in spec["expected_names"]
    }:
        score += 100
        reasons.append("expected_filename")

    if rows in spec["expected_rows"]:
        score += 50
        reasons.append(f"expected_row_count={rows}")
    else:
        nearest = min(abs(rows - expected) for expected in spec["expected_rows"])
        if nearest <= 3:
            score += 20
            reasons.append(f"near_row_count={rows}")

    keyword_hits = 0
    for keyword in spec["keywords"]:
        token = norm(keyword)
        hit = (
            token in path_text
            or any(token in column for column in column_tokens)
        )
        if hit:
            keyword_hits += 1

    score += keyword_hits * 4
    reasons.append(f"keyword_hits={keyword_hits}")

    # General methane table signals.
    for signal in [
        "label", "latitude", "longitude", "acquisition",
        "scene", "emission", "plume", "classification",
    ]:
        if any(signal in column for column in column_tokens):
            score += 2

    return score, reasons


def scan_recovery_candidates(
    search_root: Path,
    max_csv_size_mb: float,
) -> pd.DataFrame:
    records = []

    for path in search_root.rglob("*.csv"):
        try:
            size_mb = path.stat().st_size / 1024**2
            if size_mb > max_csv_size_mb:
                continue

            rows = csv_row_count(path)
            columns = file_header(path)

            for target_name in TARGETS:
                score, reasons = score_candidate(
                    target_name=target_name,
                    path=path,
                    rows=rows,
                    columns=columns,
                )
                if score < 16:
                    continue

                records.append({
                    "target_name": target_name,
                    "score": score,
                    "rows": rows,
                    "columns": len(columns),
                    "path": str(path),
                    "reasons": "|".join(reasons),
                    "column_names": "|".join(columns),
                    "size_mb": round(size_mb, 4),
                })

        except Exception as exc:
            records.append({
                "target_name": "scan_error",
                "score": 0,
                "rows": pd.NA,
                "columns": pd.NA,
                "path": str(path),
                "reasons": f"{type(exc).__name__}: {exc}",
                "column_names": "",
                "size_mb": pd.NA,
            })

    if not records:
        return pd.DataFrame(
            columns=[
                "target_name", "score", "rows", "columns", "path",
                "reasons", "column_names", "size_mb",
            ]
        )

    result = pd.DataFrame(records)
    result = result.sort_values(
        ["target_name", "score", "rows"],
        ascending=[True, False, False],
    )
    return result


def choose_candidate(
    candidates: pd.DataFrame,
    target_name: str,
    minimum_score: float = 55,
) -> Path | None:
    subset = candidates[candidates["target_name"] == target_name].copy()
    if subset.empty:
        return None

    subset = subset.sort_values(
        ["score", "rows"],
        ascending=[False, False],
    )

    best = subset.iloc[0]
    if float(best["score"]) < minimum_score:
        return None

    return Path(str(best["path"]))


def canonical_inventory(
    path: Path,
    source_name: str,
    default_sensor: str,
    default_ground_truth_type: str,
) -> pd.DataFrame:
    raw = pd.read_csv(path)
    output = pd.DataFrame(index=raw.index)

    for canonical, aliases in ALIASES.items():
        output[canonical] = first_column(raw, aliases)

    output["record_id"] = clean_text(output["record_id"])
    missing_id = output["record_id"].isna()
    output.loc[missing_id, "record_id"] = [
        f"{source_name}_{index:06d}"
        for index in output.index[missing_id]
    ]

    for column in [
        "site_id", "facility_id", "sensor", "scene_id",
        "acquisition_time_utc", "image_path",
    ]:
        output[column] = clean_text(output[column])

    output["latitude"] = pd.to_numeric(
        output["latitude"],
        errors="coerce",
    )
    output["longitude"] = pd.to_numeric(
        output["longitude"],
        errors="coerce",
    )
    output["emission_rate_kg_hr"] = pd.to_numeric(
        output["emission_rate_kg_hr"],
        errors="coerce",
    )
    output["wind_speed_m_s"] = pd.to_numeric(
        output["wind_speed_m_s"],
        errors="coerce",
    )
    output["wind_direction_deg"] = pd.to_numeric(
        output["wind_direction_deg"],
        errors="coerce",
    )

    parsed_time = pd.to_datetime(
        output["acquisition_time_utc"],
        errors="coerce",
        utc=True,
    )
    output["acquisition_time_utc"] = parsed_time.dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    output["label_raw"] = output["label"]
    output["label"] = output["label"].map(normalize_label).astype("Int64")
    output["sensor"] = output["sensor"].fillna(default_sensor)
    output["ground_truth_type"] = default_ground_truth_type
    output["ground_truth_source"] = source_name
    output["source_file"] = str(path)

    output["has_coordinates"] = (
        output["latitude"].notna()
        & output["longitude"].notna()
    )
    output["has_acquisition_time"] = (
        output["acquisition_time_utc"].notna()
    )
    output["search_ready"] = (
        output["has_coordinates"]
        & output["has_acquisition_time"]
    )

    return output


def build_search_requests(
    inventory: pd.DataFrame,
    source_name: str,
    sensors: Iterable[str],
    day_windows: Iterable[int],
) -> pd.DataFrame:
    records = []

    for _, row in inventory.iterrows():
        if not bool(row.get("search_ready", False)):
            continue

        center = pd.to_datetime(
            row["acquisition_time_utc"],
            utc=True,
            errors="coerce",
        )
        if pd.isna(center):
            continue

        for sensor in sensors:
            for days in day_windows:
                records.append({
                    "request_id": (
                        f"{source_name}|{row['record_id']}|"
                        f"{sensor}|pm{days}d"
                    ),
                    "source_record_id": row["record_id"],
                    "site_id": row.get("site_id"),
                    "facility_id": row.get("facility_id"),
                    "latitude": row.get("latitude"),
                    "longitude": row.get("longitude"),
                    "ground_truth_time_utc": row.get(
                        "acquisition_time_utc"
                    ),
                    "target_sensor": sensor,
                    "window_days": days,
                    "search_start_utc": (
                        center - pd.Timedelta(days=days)
                    ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "search_end_utc": (
                        center + pd.Timedelta(days=days)
                    ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "label": row.get("label"),
                    "ground_truth_type": row.get(
                        "ground_truth_type"
                    ),
                    "status": "pending_satellite_search",
                })

    return pd.DataFrame(records)


def build_temporal_negative_requests(
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    records = []
    offsets = [1, 3, 7, 14]

    positives = inventory[
        pd.to_numeric(inventory["label"], errors="coerce").eq(1)
    ]

    for _, row in positives.iterrows():
        center = pd.to_datetime(
            row["acquisition_time_utc"],
            utc=True,
            errors="coerce",
        )
        if pd.isna(center):
            continue
        if pd.isna(row.get("latitude")) or pd.isna(row.get("longitude")):
            continue

        for offset in offsets:
            target = center + pd.Timedelta(days=offset)
            records.append({
                "candidate_id": (
                    f"temporal_neg|{row['record_id']}|plus_{offset}d"
                ),
                "source_positive_record_id": row["record_id"],
                "site_id": row.get("site_id"),
                "facility_id": row.get("facility_id"),
                "latitude": row.get("latitude"),
                "longitude": row.get("longitude"),
                "positive_time_utc": row.get("acquisition_time_utc"),
                "days_after_positive": offset,
                "target_date_utc": target.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "target_sensor": "Sentinel-2",
                "proposed_label": 0,
                "confirmed_label": pd.NA,
                "known_release_excluded": False,
                "published_plume_excluded": False,
                "nearby_plume_excluded": False,
                "cloud_snow_qa_pass": False,
                "same_facility": True,
                "negative_validity": "pending_validation",
                "status": "pending_satellite_search",
            })

    return pd.DataFrame(records)


def main() -> int:
    args = parse_args()

    project_root = args.project_root.expanduser().resolve()
    search_root = args.search_root.expanduser().resolve()
    outputs = project_root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)

    master_path = (
        args.master.expanduser().resolve()
        if args.master
        else outputs / "002_unified_methane_master_dedup.csv"
    )

    if not master_path.exists():
        raise FileNotFoundError(f"Master table not found: {master_path}")

    master = pd.read_csv(master_path)
    fixed_master = fix_landsat_master(master)

    fixed_master_path = outputs / "012_unified_methane_master_fixed.csv"
    fixed_master.to_csv(fixed_master_path, index=False)

    fixed_counts_path = outputs / "013_unified_methane_counts_fixed.csv"
    write_fixed_counts(fixed_master, fixed_counts_path)

    candidates = scan_recovery_candidates(
        search_root=search_root,
        max_csv_size_mb=args.max_csv_size_mb,
    )
    candidate_path = outputs / "014_dataset_recovery_candidates.csv"
    candidates.to_csv(candidate_path, index=False)

    summary_records = []
    chosen: dict[str, Path | None] = {}

    for target_name in TARGETS:
        candidate = choose_candidate(candidates, target_name)
        chosen[target_name] = candidate

        target_subset = candidates[
            candidates["target_name"] == target_name
        ]
        summary_records.append({
            "target_name": target_name,
            "chosen_path": str(candidate) if candidate else "",
            "chosen": candidate is not None,
            "top_score": (
                float(target_subset["score"].max())
                if not target_subset.empty else pd.NA
            ),
            "candidate_count": len(target_subset),
        })

    summary = pd.DataFrame(summary_records)
    summary_path = outputs / "015_dataset_recovery_summary.csv"
    summary.to_csv(summary_path, index=False)

    methaneair_candidate = (
        chosen["methaneair_observations_435"]
        or chosen["methaneair_baseline_s2_110"]
    )

    if methaneair_candidate:
        inventory = canonical_inventory(
            path=methaneair_candidate,
            source_name="MethaneAIR",
            default_sensor="MethaneAIR",
            default_ground_truth_type="published_detection_or_catalog",
        )
        inventory_path = (
            outputs / "016_methaneair_observation_inventory.csv"
        )
        inventory.to_csv(inventory_path, index=False)

        requests = build_search_requests(
            inventory=inventory,
            source_name="MethaneAIR",
            sensors=["Sentinel-2"],
            day_windows=[0, 1, 3],
        )
        requests_path = outputs / "017_methaneair_s2_search_requests.csv"
        requests.to_csv(requests_path, index=False)

        negatives = build_temporal_negative_requests(inventory)
        negatives_path = (
            outputs / "018_methaneair_temporal_negative_requests.csv"
        )
        negatives.to_csv(negatives_path, index=False)

    carbon_candidate = chosen["carbonmapper_observations_226"]
    if carbon_candidate:
        inventory = canonical_inventory(
            path=carbon_candidate,
            source_name="Carbon Mapper",
            default_sensor="Carbon Mapper",
            default_ground_truth_type="provider_observation_classification",
        )
        inventory_path = (
            outputs / "019_carbonmapper_observation_inventory.csv"
        )
        inventory.to_csv(inventory_path, index=False)

        requests = build_search_requests(
            inventory=inventory,
            source_name="CarbonMapper",
            sensors=["Sentinel-2", "Landsat-8/9"],
            day_windows=[0, 1, 3],
        )
        requests_path = (
            outputs / "020_carbonmapper_satellite_search_requests.csv"
        )
        requests.to_csv(requests_path, index=False)

    historical_candidate = chosen["historical_multisatellite_17"]
    if historical_candidate:
        inventory = canonical_inventory(
            path=historical_candidate,
            source_name="Historical multisatellite",
            default_sensor="multiple",
            default_ground_truth_type="provider_acquisition_classification",
        )
        inventory_path = (
            outputs / "021_historical_multisatellite_inventory.csv"
        )
        inventory.to_csv(inventory_path, index=False)

    landsat = fixed_master[
        fixed_master.get("model_family", pd.Series(dtype=str))
        .astype("string")
        .eq("landsat_temporal")
    ]

    print("=" * 80)
    print("Expansion batch complete")
    print("Fixed master:", fixed_master_path)
    print("Fixed counts:", fixed_counts_path)
    print("Recovery candidates:", candidate_path)
    print("Recovery summary:", summary_path)
    print()
    print("Landsat rows:", len(landsat))
    print(
        "Landsat rows with acquisition time:",
        int(landsat["acquisition_time_utc"].notna().sum())
        if "acquisition_time_utc" in landsat.columns else 0,
    )
    print(
        "Landsat unique events:",
        landsat["event_group_id"].nunique(dropna=True)
        if "event_group_id" in landsat.columns else 0,
    )
    print()
    print(summary.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
