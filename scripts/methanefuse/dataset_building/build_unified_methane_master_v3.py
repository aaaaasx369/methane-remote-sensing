#!/usr/bin/env python3
"""
Build one unified methane master table from the project's existing CSV tables.

The script does three things:
1. Inventories candidate CSV files and their schemas.
2. Converts known/likely source tables to one canonical row format.
3. Deduplicates rows representing the same sensor acquisition while preserving
   source-membership flags and provenance.

Important:
- One row in the deduplicated table is one site/time/sensor acquisition.
- Derived evaluation tables are merged as membership flags, not counted as new
  ground-truth observations when their acquisition key matches another row.
- "model_ready" means the row has a binary label and at least one usable local
  model asset. It does not mean different sensors can be fed into one identical
  neural-network input without an adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


CANONICAL_COLUMNS = [
    "master_id",
    "dedup_key",
    "event_group_id",
    "source_datasets",
    "source_row_ids",
    "record_role",
    "site_id",
    "facility_id",
    "campaign_id",
    "latitude",
    "longitude",
    "sensor",
    "platform",
    "instrument",
    "scene_id",
    "acquisition_time_utc",
    "label",
    "label_text",
    "ground_truth_type",
    "ground_truth_source",
    "label_confidence",
    "controlled_release",
    "emission_rate_kg_hr",
    "release_start_utc",
    "release_end_utc",
    "wind_speed_m_s",
    "wind_direction_deg",
    "cloud_fraction",
    "snow_fraction",
    "image_path",
    "t0_path",
    "t90_path",
    "t360_path",
    "con_tif_path",
    "plume_tif_path",
    "rgb_tif_path",
    "qa_pass",
    "qa_metric",
    "qa_threshold",
    "model_family",
    "model_ready",
    "model_ready_reason",
    "dataset_role",
    "in_s2_baseline",
    "in_five_site_eval",
    "in_exact_s2_eval",
    "in_landsat_eval",
    "in_carbonmapper_eval",
    "duplicate_source_count",
]


ALIASES: dict[str, list[str]] = {
    "event_group_id": [
        "event_group_id", "event_id", "release_id", "observation_id",
        "controlled_release_id", "group_id",
    ],
    "site_id": [
        "site_id", "site", "location_id", "site_name", "facility_site",
        "source_site", "master_site",
    ],
    "facility_id": [
        "facility_id", "facility", "source_id", "asset_id", "facility_name",
    ],
    "campaign_id": [
        "campaign_id", "campaign", "study", "experiment", "source_campaign",
    ],
    "latitude": [
        "latitude", "lat", "source_latitude", "plume_latitude",
        "release_latitude", "center_lat",
    ],
    "longitude": [
        "longitude", "lon", "lng", "source_longitude", "plume_longitude",
        "release_longitude", "center_lon",
    ],
    "sensor": [
        "sensor", "satellite", "landsat_sensor", "sensor_name",
        "instrument_sensor",
    ],
    "platform": ["platform", "satellite_platform", "aircraft", "mission"],
    "instrument": ["instrument", "instrument_name", "sensor_code"],
    "scene_id": [
        "scene_id", "source_scene_id", "image_id", "product_id",
        "granule_id", "system_index", "landsat_scene_id", "s2_scene_id",
        "scene",
    ],
    "acquisition_time_utc": [
        "acquisition_time_utc", "source_acquisition_time_utc",
        "acquisition_time", "scene_timestamp", "datetime", "timestamp",
        "image_time", "overpass_time", "satellite_time", "date_time",
        "date",
    ],
    "label": [
        "label", "ground_truth", "target", "y_true", "class",
        "binary_label", "is_positive",
    ],
    "label_text": [
        "label_text", "classification", "tc_classification", "result",
        "ground_truth_class", "status_label",
    ],
    "ground_truth_type": [
        "ground_truth_type", "truth_type", "label_type",
        "performance_interpretation", "observation_type",
    ],
    "ground_truth_source": [
        "ground_truth_source", "label_provenance", "truth_source",
        "label_source", "data_source", "source_dataset",
    ],
    "label_confidence": [
        "label_confidence", "confidence", "ground_truth_confidence",
        "quality_label",
    ],
    "controlled_release": [
        "controlled_release", "controlled_release_verified",
        "is_controlled_release", "controlled",
    ],
    "emission_rate_kg_hr": [
        "emission_rate_kg_hr", "emission_kg_hr",
        "consensus_release_rate_kg_h", "metered_release_rate_kg_hr",
        "release_rate_kg_hr", "ground_truth_rate_kg_hr", "emission_auto",
        "emission_rate", "release_rate",
    ],
    "release_start_utc": [
        "release_start_utc", "release_start", "start_time", "release_start_time",
    ],
    "release_end_utc": [
        "release_end_utc", "release_end", "end_time", "release_end_time",
    ],
    "wind_speed_m_s": [
        "wind_speed_m_s", "wind_speed", "wind_speed_avg_auto",
        "wind_m_s",
    ],
    "wind_direction_deg": [
        "wind_direction_deg", "wind_direction", "wind_direction_avg_auto",
        "wind_dir_deg",
    ],
    "cloud_fraction": [
        "cloud_fraction", "cloud_cover", "cloud_probability", "clear_fraction",
        "scl_clear_fraction",
    ],
    "snow_fraction": ["snow_fraction", "snow_cover", "snow_probability"],
    "image_path": [
        "image_path", "patch_path", "tif_path", "local_path",
        "raster_path", "file_path",
    ],
    "t0_path": [
        "t0_path", "s2_0_path", "l89_0_path", "image_t0",
        "path_t0", "t0", "t0_tif",
    ],
    "t90_path": [
        "t90_path", "s2_90_path", "l89_90_path", "image_t90",
        "path_t90", "t90", "t90_tif",
    ],
    "t360_path": [
        "t360_path", "s2_360_path", "l89_360_path", "image_t360",
        "path_t360", "t360", "t360_tif",
    ],
    "con_tif_path": [
        "local_con_tif", "con_tif_path", "con_tif", "concentration_tif",
    ],
    "plume_tif_path": [
        "local_plume_tif", "plume_tif_path", "plume_tif",
    ],
    "rgb_tif_path": [
        "local_rgb_tif", "rgb_tif_path", "rgb_tif",
    ],
    "qa_pass": [
        "qa_pass", "all_tiffs_valid", "quality_pass", "usable",
        "final_flag", "is_usable", "passes_qa",
    ],
    "qa_metric": [
        "qa_metric", "minimum_scl_clear_fraction",
        "minimum_scl_clear_fraction_qa", "minimum_qa_clear_fraction",
        "clear_fraction", "scl_clear_fraction", "valid_fraction",
        "qa_fraction",
    ],
    "qa_threshold": ["qa_threshold", "minimum_clear", "min_clear"],
    "dataset_role": ["dataset_role", "role", "split_role"],
}


@dataclass(frozen=True)
class SourceSpec:
    name: str
    paths: tuple[str, ...]
    record_role: str
    default_sensor: str | None = None
    default_model_family: str | None = None
    default_ground_truth_source: str | None = None
    default_ground_truth_type: str | None = None
    default_label_confidence: str | None = None
    default_controlled_release: bool | None = None
    membership_flag: str | None = None


SOURCE_SPECS = [
    SourceSpec(
        name="methaneair_s2_dataset",
        paths=("outputs/18_methaneair_s2_dataset_table.csv",),
        record_role="sensor_acquisition",
        default_sensor="Sentinel-2",
        default_model_family="s2_single",
        default_ground_truth_source="MethaneAIR published plume",
        default_ground_truth_type="published_positive",
        default_label_confidence="medium",
        default_controlled_release=False,
        membership_flag="in_s2_baseline",
    ),
    SourceSpec(
        name="controlled_release_s2",
        paths=("outputs/20_controlled_release_s2_patch_index.csv",),
        record_role="sensor_acquisition",
        default_sensor="Sentinel-2",
        default_model_family="s2_single",
        default_ground_truth_source="controlled release log",
        default_ground_truth_type="controlled_release_binary",
        default_label_confidence="high",
        default_controlled_release=True,
        membership_flag="in_s2_baseline",
    ),
    SourceSpec(
        name="five_site_manifest",
        paths=(
            "data/custom/five_site_zero_shot_eval.csv",
            "data/custom/five_site_zero_shot_eval_scl80.csv",
        ),
        record_role="sensor_acquisition",
        default_sensor="Sentinel-2",
        default_model_family="s2_temporal",
        default_ground_truth_source="five-site multisource manifest",
        default_ground_truth_type="mixed_binary",
        default_label_confidence="high",
        membership_flag="in_five_site_eval",
    ),
    SourceSpec(
        name="exact_methaneair_s2_eval",
        paths=(
            "data/custom/methaneair_s2_p1_zero_shot_eval_final16.csv",
        ),
        record_role="sensor_acquisition",
        default_sensor="Sentinel-2",
        default_model_family="s2_temporal",
        default_ground_truth_source="MethaneAIR published plume",
        default_ground_truth_type="published_positive",
        default_label_confidence="medium",
        default_controlled_release=False,
        membership_flag="in_exact_s2_eval",
    ),
    SourceSpec(
        name="landsat89_eval",
        paths=(
            "data/custom/landsat89_high_emission_zero_shot_eval.csv",
            "data/custom/landsat89_high_emission_zero_shot_eval_qa80.csv",
        ),
        record_role="sensor_acquisition",
        default_sensor="Landsat-8/9",
        default_model_family="landsat_temporal",
        default_ground_truth_source="controlled release / matched benchmark",
        default_ground_truth_type="binary",
        default_label_confidence="high",
        membership_flag="in_landsat_eval",
    ),
    SourceSpec(
        name="carbonmapper_2026_raster",
        paths=(
            "../data/carbon_mapper_casa_grande_2026/carbon_mapper_2026_master.csv",
        ),
        record_role="sensor_acquisition",
        default_sensor="Carbon Mapper Tanager",
        default_model_family="carbonmapper_concentration",
        default_ground_truth_source="Carbon Mapper published plume",
        default_ground_truth_type="published_positive",
        default_label_confidence="medium",
        default_controlled_release=False,
        membership_flag="in_carbonmapper_eval",
    ),
]


AUTO_DISCOVERY_PATTERNS = {
    "carbonmapper_observation": [
        "outputs/**/*carbon*mapper*observation*.csv",
        "outputs/**/*carbon*mapper*classification*.csv",
        "outputs/**/*carbon*mapper*result*.csv",
    ],
    "historical_multisatellite": [
        "outputs/**/*multisatellite*.csv",
        "outputs/**/*multi*satellite*.csv",
        "outputs/**/*historical*satellite*.csv",
    ],
    "methaneair_inventory": [
        "outputs/**/*methaneair*inventory*.csv",
        "outputs/**/*methaneair*catalog*.csv",
        "outputs/**/*methaneair*master*.csv",
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
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <project-root>/outputs.",
    )
    return parser.parse_args()


def normalize_name(name: Any) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    used: dict[str, int] = {}
    for col in df.columns:
        base = normalize_name(col)
        if not base:
            base = "unnamed"
        count = used.get(base, 0)
        used[base] = count + 1
        renamed[col] = base if count == 0 else f"{base}_{count+1}"
    return df.rename(columns=renamed)


def first_series(df: pd.DataFrame, names: Iterable[str]) -> pd.Series:
    for name in names:
        normalized = normalize_name(name)
        if normalized in df.columns:
            return df[normalized]
    return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")


def normalize_bool(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isnan(value) if isinstance(value, float) else False:
            return pd.NA
        return bool(int(value))
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "pass", "passed", "usable"}:
        return True
    if text in {"0", "false", "no", "n", "fail", "failed", "unusable"}:
        return False
    return pd.NA


def normalize_label(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            number = int(value)
        except Exception:
            return pd.NA
        return number if number in (0, 1) else pd.NA
    text = str(value).strip().lower()
    positives = {
        "1", "positive", "pos", "plume", "detected", "tp", "true positive",
        "release_on", "yes", "p",
    }
    negatives = {
        "0", "negative", "neg", "no plume", "not detected", "tn",
        "true negative", "release_off", "no", "n",
    }
    if text in positives:
        return 1
    if text in negatives:
        return 0
    return pd.NA


def text_or_na(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    return out.mask(out.isin(["", "nan", "None", "<NA>"]), pd.NA)


def numeric_or_na(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def time_or_text(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    output = parsed.dt.strftime("%Y-%m-%dT%H:%M:%SZ").astype("string")
    original = text_or_na(series)
    return output.fillna(original)


def existing_local_path(value: Any, project_root: Path) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip()
    if not text or text.lower().startswith(("http://", "https://")):
        return False
    path = Path(text).expanduser()
    if not path.is_absolute():
        candidate_paths = [
            project_root / path,
            project_root.parent / path,
            Path.cwd() / path,
        ]
    else:
        candidate_paths = [path]
    return any(candidate.exists() for candidate in candidate_paths)


def choose_source_row_id(df: pd.DataFrame) -> pd.Series:
    candidates = [
        "sample_id", "plume_id", "observation_id", "event_id", "scene_id",
        "id", "record_id", "source_row_id",
    ]
    for name in candidates:
        if name in df.columns:
            value = text_or_na(df[name])
            if value.notna().any():
                return value
    return pd.Series([str(i) for i in range(len(df))], index=df.index, dtype="string")


def resolve_paths(project_root: Path, patterns: Iterable[str]) -> list[Path]:
    found: list[Path] = []
    for pattern in patterns:
        if any(ch in pattern for ch in "*?[]"):
            found.extend(project_root.glob(pattern))
        else:
            path = (project_root / pattern).resolve()
            if path.exists():
                found.append(path)
    unique = []
    seen = set()
    for path in found:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def infer_model_family(source_name: str, sensor: pd.Series) -> pd.Series:
    default = pd.Series([pd.NA] * len(sensor), index=sensor.index, dtype="string")
    lower = sensor.astype("string").str.lower()
    default = default.mask(lower.str.contains("sentinel", na=False), "s2_single")
    default = default.mask(lower.str.contains("landsat", na=False), "landsat_single")
    default = default.mask(
        lower.str.contains("carbon mapper|tanager", na=False),
        "carbonmapper_concentration",
    )
    if "historical" in source_name:
        default = default.fillna("observation_only")
    return default


def infer_label_from_text(label: pd.Series, label_text: pd.Series) -> pd.Series:
    normalized = label.map(normalize_label)
    missing = normalized.isna()
    if missing.any():
        normalized.loc[missing] = label_text.loc[missing].map(normalize_label)
    return normalized.astype("Int64")


def model_ready_status(
    row: pd.Series,
    project_root: Path,
) -> tuple[bool, str]:
    if pd.isna(row.get("label")):
        return False, "missing_binary_label"

    model_family = str(row.get("model_family") or "")
    paths = {
        key: row.get(key)
        for key in [
            "image_path", "t0_path", "t90_path", "t360_path",
            "con_tif_path", "plume_tif_path", "rgb_tif_path",
        ]
    }

    if model_family in {"s2_temporal", "landsat_temporal"}:
        required = ["t0_path", "t90_path", "t360_path"]
        if all(existing_local_path(paths[key], project_root) for key in required):
            return True, "temporal_stack_complete"
        return False, "missing_temporal_stack"

    if model_family == "carbonmapper_concentration":
        if existing_local_path(paths["con_tif_path"], project_root):
            return True, "concentration_raster_available"
        return False, "missing_concentration_raster"

    if existing_local_path(paths["image_path"], project_root):
        return True, "single_image_available"

    if existing_local_path(paths["t0_path"], project_root):
        return True, "t0_image_available"

    return False, "no_local_model_asset"


def build_canonical(
    raw_df: pd.DataFrame,
    source_path: Path,
    spec: SourceSpec,
    project_root: Path,
) -> pd.DataFrame:
    df = normalize_columns(raw_df.copy())
    out = pd.DataFrame(index=df.index)

    for canonical, aliases in ALIASES.items():
        out[canonical] = first_series(df, aliases)

    out["source_datasets"] = spec.name
    source_row_id = choose_source_row_id(df)
    out["source_row_ids"] = source_row_id
    out["record_role"] = spec.record_role

    for col in [
        "event_group_id", "site_id", "facility_id", "campaign_id",
        "sensor", "platform", "instrument", "scene_id", "label_text",
        "ground_truth_type", "ground_truth_source", "label_confidence",
        "image_path", "t0_path", "t90_path", "t360_path",
        "con_tif_path", "plume_tif_path", "rgb_tif_path",
        "qa_metric", "qa_threshold", "dataset_role",
    ]:
        out[col] = text_or_na(out[col])

    out["latitude"] = numeric_or_na(out["latitude"])
    out["longitude"] = numeric_or_na(out["longitude"])
    out["emission_rate_kg_hr"] = numeric_or_na(out["emission_rate_kg_hr"])
    out["wind_speed_m_s"] = numeric_or_na(out["wind_speed_m_s"])
    out["wind_direction_deg"] = numeric_or_na(out["wind_direction_deg"])
    out["cloud_fraction"] = numeric_or_na(out["cloud_fraction"])
    out["snow_fraction"] = numeric_or_na(out["snow_fraction"])

    out["acquisition_time_utc"] = time_or_text(out["acquisition_time_utc"])
    out["release_start_utc"] = time_or_text(out["release_start_utc"])
    out["release_end_utc"] = time_or_text(out["release_end_utc"])

    out["label"] = infer_label_from_text(out["label"], out["label_text"])
    out["qa_pass"] = out["qa_pass"].map(normalize_bool).astype("boolean")

    # Curated subset filenames explicitly represent quality-approved rows.
    source_name_lower = source_path.name.lower()
    curated_quality_subset = (
        "scl80" in source_name_lower
        or "qa80" in source_name_lower
        or "final16" in source_name_lower
        or spec.name == "carbonmapper_2026_raster"
    )
    if curated_quality_subset:
        out["qa_pass"] = out["qa_pass"].fillna(True)

    out["controlled_release"] = (
        out["controlled_release"].map(normalize_bool).astype("boolean")
    )

    if spec.default_sensor:
        out["sensor"] = out["sensor"].fillna(spec.default_sensor)
    if spec.default_ground_truth_source:
        out["ground_truth_source"] = out["ground_truth_source"].fillna(
            spec.default_ground_truth_source
        )
    if spec.default_ground_truth_type:
        out["ground_truth_type"] = out["ground_truth_type"].fillna(
            spec.default_ground_truth_type
        )
    if spec.default_label_confidence:
        out["label_confidence"] = out["label_confidence"].fillna(
            spec.default_label_confidence
        )
    if spec.default_controlled_release is not None:
        out["controlled_release"] = out["controlled_release"].fillna(
            spec.default_controlled_release
        )

    if spec.default_model_family:
        out["model_family"] = spec.default_model_family
    else:
        out["model_family"] = infer_model_family(spec.name, out["sensor"])

    for flag in [
        "in_s2_baseline", "in_five_site_eval", "in_exact_s2_eval",
        "in_landsat_eval", "in_carbonmapper_eval",
    ]:
        out[flag] = False
    if spec.membership_flag:
        out[spec.membership_flag] = True

    out["dataset_role"] = out["dataset_role"].fillna(
        "model_candidate" if spec.record_role == "sensor_acquisition"
        else "observation_inventory"
    )

    # If an obvious local image path was not captured, inspect remaining columns.
    if out["image_path"].isna().all():
        path_like = [
            col for col in df.columns
            if any(token in col for token in ["path", "tif", "image", "file"])
        ]
        for col in path_like:
            candidate = text_or_na(df[col])
            fill_mask = out["image_path"].isna() & candidate.notna()
            out.loc[fill_mask, "image_path"] = candidate.loc[fill_mask]

    # Keep the originating path in provenance.
    out["source_datasets"] = (
        out["source_datasets"] + "::" + str(source_path)
    )

    # Stable event grouping where possible.
    fallback_event = (
        out["site_id"].fillna("unknown_site").astype("string")
        + "|"
        + out["acquisition_time_utc"].fillna("unknown_time").astype("string")
    )
    out["event_group_id"] = out["event_group_id"].fillna(fallback_event)

    # Dedup only when acquisition identity is sufficiently specific.
    identity_parts = pd.DataFrame({
        "sensor": out["sensor"].fillna("unknown_sensor").astype("string"),
        "site": out["site_id"].fillna(out["facility_id"]).fillna("unknown_site").astype("string"),
        "scene": out["scene_id"].fillna("unknown_scene").astype("string"),
        "time": out["acquisition_time_utc"].fillna("unknown_time").astype("string"),
    })

    sufficiently_specific = (
        (identity_parts["scene"] != "unknown_scene")
        | (identity_parts["time"] != "unknown_time")
    )

    # Preserve separate model patches/plumes within the same acquisition.
    # The same sample_id/plume_id appearing in raw and QA tables still merges.
    entity_id = source_row_id.fillna("unknown_entity").astype("string")

    generic_key = (
        identity_parts["sensor"] + "|"
        + identity_parts["site"] + "|"
        + identity_parts["scene"] + "|"
        + identity_parts["time"] + "|"
        + entity_id
    )

    unique_fallback = spec.name + "|" + entity_id

    out["dedup_key"] = generic_key.where(
        sufficiently_specific,
        unique_fallback,
    )

    statuses = out.apply(
        lambda row: model_ready_status(row, project_root),
        axis=1,
        result_type="expand",
    )
    out["model_ready"] = statuses[0].astype(bool)
    out["model_ready_reason"] = statuses[1].astype("string")

    out["duplicate_source_count"] = 1
    out["master_id"] = out["dedup_key"].map(
        lambda x: "m_" + hashlib.sha1(str(x).encode("utf-8")).hexdigest()[:16]
    )

    for col in CANONICAL_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    return out[CANONICAL_COLUMNS]


def first_non_null(values: pd.Series) -> Any:
    cleaned = values.dropna()
    if cleaned.empty:
        return pd.NA
    for value in cleaned:
        if isinstance(value, str) and value.strip():
            return value
        if not isinstance(value, str):
            return value
    return cleaned.iloc[0]


def join_unique(values: pd.Series) -> str:
    items = []
    seen = set()
    for value in values.dropna():
        for token in str(value).split("|"):
            token = token.strip()
            if token and token not in seen:
                seen.add(token)
                items.append(token)
    return "|".join(items)


def merge_group(group: pd.DataFrame, project_root: Path) -> pd.Series:
    merged: dict[str, Any] = {}

    join_cols = ["source_datasets", "source_row_ids"]
    bool_or_cols = [
        "in_s2_baseline", "in_five_site_eval", "in_exact_s2_eval",
        "in_landsat_eval", "in_carbonmapper_eval", "model_ready",
    ]

    for col in CANONICAL_COLUMNS:
        if col in join_cols:
            merged[col] = join_unique(group[col])
        elif col in bool_or_cols:
            merged[col] = bool(group[col].fillna(False).astype(bool).any())
        elif col == "duplicate_source_count":
            merged[col] = len(group)
        else:
            merged[col] = first_non_null(group[col])

    # Prefer high-confidence labels where duplicates disagree.
    labeled = group[group["label"].notna()].copy()
    if not labeled.empty:
        confidence_rank = {"high": 3, "medium": 2, "low": 1}
        labeled["_rank"] = (
            labeled["label_confidence"]
            .astype("string")
            .str.lower()
            .map(confidence_rank)
            .fillna(0)
        )
        best = labeled.sort_values("_rank", ascending=False).iloc[0]
        for col in [
            "label", "label_text", "ground_truth_type",
            "ground_truth_source", "label_confidence", "controlled_release",
            "emission_rate_kg_hr", "release_start_utc", "release_end_utc",
        ]:
            if pd.notna(best.get(col)):
                merged[col] = best[col]

    merged_series = pd.Series(merged)
    ready, reason = model_ready_status(merged_series, project_root)
    merged["model_ready"] = bool(ready)
    merged["model_ready_reason"] = reason

    return pd.Series(merged)


def inventory_csv(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else pd.NA,
        "rows": pd.NA,
        "columns": pd.NA,
        "column_names": "",
        "label_counts": "",
        "error": "",
    }
    if not path.exists():
        return record

    try:
        df = pd.read_csv(path)
        normalized = normalize_columns(df)
        record["rows"] = len(df)
        record["columns"] = len(df.columns)
        record["column_names"] = "|".join(map(str, df.columns))

        label_series = first_series(normalized, ALIASES["label"])
        label_text = first_series(normalized, ALIASES["label_text"])
        labels = infer_label_from_text(label_series, label_text)
        counts = labels.value_counts(dropna=False).to_dict()
        record["label_counts"] = json.dumps(
            {str(key): int(value) for key, value in counts.items()},
            ensure_ascii=False,
        )
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"

    return record


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else project_root / "outputs"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if not project_root.exists():
        raise FileNotFoundError(f"Project root not found: {project_root}")

    inventory_paths: list[Path] = []
    canonical_frames: list[pd.DataFrame] = []
    schema_records: list[dict[str, Any]] = []

    for spec in SOURCE_SPECS:
        paths = resolve_paths(project_root, spec.paths)
        inventory_paths.extend(paths)

        for path in paths:
            try:
                raw = pd.read_csv(path)
                canonical = build_canonical(
                    raw_df=raw,
                    source_path=path,
                    spec=spec,
                    project_root=project_root,
                )
                canonical_frames.append(canonical)
                schema_records.append({
                    "source_name": spec.name,
                    "path": str(path),
                    "status": "loaded",
                    "rows": len(raw),
                    "unmapped_columns": "|".join(
                        col for col in normalize_columns(raw).columns
                        if not any(
                            col in [normalize_name(x) for x in aliases]
                            for aliases in ALIASES.values()
                        )
                    ),
                    "error": "",
                })
            except Exception as exc:
                schema_records.append({
                    "source_name": spec.name,
                    "path": str(path),
                    "status": "failed",
                    "rows": pd.NA,
                    "unmapped_columns": "",
                    "error": f"{type(exc).__name__}: {exc}",
                })

    # Auto-discover observation-only tables. These are included in the same
    # master table, but will normally remain model_ready=False until matched
    # with a local image asset.
    for source_name, patterns in AUTO_DISCOVERY_PATTERNS.items():
        paths = resolve_paths(project_root, patterns)
        inventory_paths.extend(paths)

        if source_name == "carbonmapper_observation":
            spec = SourceSpec(
                name=source_name,
                paths=tuple(),
                record_role="observation",
                default_sensor="Carbon Mapper",
                default_model_family="observation_only",
                default_ground_truth_source="Carbon Mapper observation table",
                default_ground_truth_type="provider_classification",
                default_label_confidence="high",
            )
        elif source_name == "historical_multisatellite":
            spec = SourceSpec(
                name=source_name,
                paths=tuple(),
                record_role="observation",
                default_model_family="observation_only",
                default_ground_truth_source="historical multisatellite study",
                default_ground_truth_type="provider_classification",
                default_label_confidence="high",
            )
        else:
            spec = SourceSpec(
                name=source_name,
                paths=tuple(),
                record_role="observation",
                default_sensor="MethaneAIR",
                default_model_family="observation_only",
                default_ground_truth_source="MethaneAIR catalog",
                default_ground_truth_type="published_positive",
                default_label_confidence="medium",
                default_controlled_release=False,
            )

        for path in paths:
            # Avoid reloading files already loaded by an explicit spec.
            if any(str(path) in frame["source_datasets"].astype(str).iloc[0]
                   for frame in canonical_frames if not frame.empty):
                continue
            try:
                raw = pd.read_csv(path)
                canonical = build_canonical(
                    raw_df=raw,
                    source_path=path,
                    spec=spec,
                    project_root=project_root,
                )
                canonical_frames.append(canonical)
                schema_records.append({
                    "source_name": spec.name,
                    "path": str(path),
                    "status": "loaded",
                    "rows": len(raw),
                    "unmapped_columns": "",
                    "error": "",
                })
            except Exception as exc:
                schema_records.append({
                    "source_name": spec.name,
                    "path": str(path),
                    "status": "failed",
                    "rows": pd.NA,
                    "unmapped_columns": "",
                    "error": f"{type(exc).__name__}: {exc}",
                })

    unique_inventory_paths = sorted(set(inventory_paths))
    inventory_df = pd.DataFrame(
        [inventory_csv(path) for path in unique_inventory_paths]
    )
    inventory_path = output_dir / "000_source_file_inventory.csv"
    inventory_df.to_csv(inventory_path, index=False)

    schema_df = pd.DataFrame(schema_records)
    schema_path = output_dir / "004_schema_mapping_report.csv"
    schema_df.to_csv(schema_path, index=False)

    if not canonical_frames:
        print("No source tables were loaded.")
        print("Inventory:", inventory_path)
        print("Schema report:", schema_path)
        return 1

    raw_master = pd.concat(canonical_frames, ignore_index=True)
    raw_path = output_dir / "001_unified_methane_master_raw.csv"
    raw_master.to_csv(raw_path, index=False)

    # Explicit group iteration avoids pandas groupby.apply removing
    # the grouping column in newer pandas versions.
    merged_rows: list[pd.Series] = []

    for dedup_key, group in raw_master.groupby(
        "dedup_key",
        dropna=False,
        sort=False,
    ):
        group = group.copy()
        group["dedup_key"] = dedup_key

        merged = merge_group(group, project_root)
        merged["dedup_key"] = dedup_key
        merged_rows.append(merged)

    dedup_master = pd.DataFrame(merged_rows)

    if dedup_master.empty:
        dedup_master = pd.DataFrame(columns=CANONICAL_COLUMNS)

    for col in CANONICAL_COLUMNS:
        if col not in dedup_master.columns:
            dedup_master[col] = pd.NA

    dedup_master = dedup_master[CANONICAL_COLUMNS]

    # Recreate a stable master ID after merging.
    dedup_master["master_id"] = dedup_master["dedup_key"].map(
        lambda x: "m_" + hashlib.sha1(
            str(x).encode("utf-8")
        ).hexdigest()[:16]
    )

    dedup_path = output_dir / "002_unified_methane_master_dedup.csv"
    dedup_master.to_csv(dedup_path, index=False)

    counts_records = []

    def add_count(group_type: str, group_value: str, subset: pd.DataFrame) -> None:
        labels = subset["label"].value_counts(dropna=False).to_dict()
        counts_records.append({
            "group_type": group_type,
            "group_value": group_value,
            "rows": len(subset),
            "unique_events": subset["event_group_id"].nunique(dropna=True),
            "model_ready": int(subset["model_ready"].fillna(False).sum()),
            "qa_pass": int(subset["qa_pass"].eq(True).sum()),
            "qa_model_ready": int(
                (
                    subset["model_ready"].fillna(False)
                    & subset["qa_pass"].eq(True)
                ).sum()
            ),
            "positive": int(labels.get(1, 0)),
            "negative": int(labels.get(0, 0)),
            "unlabeled": int(subset["label"].isna().sum()),
        })

    add_count("all", "all", dedup_master)
    for value, subset in dedup_master.groupby(
        dedup_master["sensor"].fillna("missing_sensor"),
        dropna=False,
    ):
        add_count("sensor", str(value), subset)
    for value, subset in dedup_master.groupby(
        dedup_master["model_family"].fillna("missing_model_family"),
        dropna=False,
    ):
        add_count("model_family", str(value), subset)
    for value, subset in dedup_master.groupby(
        dedup_master["ground_truth_source"].fillna("missing_ground_truth_source"),
        dropna=False,
    ):
        add_count("ground_truth_source", str(value), subset)

    counts_df = pd.DataFrame(counts_records)
    counts_path = output_dir / "003_unified_methane_counts.csv"
    counts_df.to_csv(counts_path, index=False)

    duplicates = dedup_master[
        dedup_master["duplicate_source_count"].fillna(1).astype(int) > 1
    ].copy()
    duplicate_path = output_dir / "005_duplicate_merge_audit.csv"
    duplicates.to_csv(duplicate_path, index=False)

    print("=" * 78)
    print("Unified methane master build complete")
    print("Project root:", project_root)
    print("Loaded source tables:", len(canonical_frames))
    print("Raw rows:", len(raw_master))
    print("Deduplicated rows:", len(dedup_master))
    print("Model-ready rows:", int(dedup_master["model_ready"].sum()))
    print("Positive rows:", int((dedup_master["label"] == 1).sum()))
    print("Negative rows:", int((dedup_master["label"] == 0).sum()))
    print("Unlabeled rows:", int(dedup_master["label"].isna().sum()))
    print()
    print("Outputs:")
    print(" ", inventory_path)
    print(" ", raw_path)
    print(" ", dedup_path)
    print(" ", counts_path)
    print(" ", schema_path)
    print(" ", duplicate_path)
    print()
    print("Recommended source of truth:")
    print(" ", dedup_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
