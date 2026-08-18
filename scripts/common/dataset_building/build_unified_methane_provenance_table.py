from __future__ import annotations

from pathlib import Path
import json
import re
import pandas as pd


PROJECT = Path("/Users/happydoraaa/methane_release_project")
OUTPUTS = PROJECT / "outputs"

S2_PATH = OUTPUTS / "36_multisite_s2_master_table.csv"
METHANEAIR_PATH = OUTPUTS / "67_all45_ground_truth_nc_mapping.csv"

OUT_CSV = OUTPUTS / "69_unified_methane_provenance_table.csv"
OUT_AUDIT = OUTPUTS / "69_unified_methane_provenance_audit.txt"


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {str(column).lower(): str(column) for column in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def get_series(
    df: pd.DataFrame,
    candidates: list[str],
    default=pd.NA,
) -> pd.Series:
    column = first_existing_column(df, candidates)

    if column is None:
        return pd.Series([default] * len(df), index=df.index)

    return df[column]


def normalize_time(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    return parsed.dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def find_carbon_mapper_file() -> tuple[Path | None, list[dict]]:
    candidates = []

    for path in sorted(OUTPUTS.rglob("*.csv")):
        if path in {S2_PATH, METHANEAIR_PATH, OUT_CSV}:
            continue

        try:
            header = pd.read_csv(path, nrows=0)
        except Exception:
            continue

        columns = [str(column) for column in header.columns]
        lower = {column.lower() for column in columns}

        score = 0

        for token in [
            "tc_classification",
            "detection",
            "ground_truth_rate_kg_hr",
            "emission_rate",
            "source_row_count",
            "satellite",
        ]:
            if token in lower:
                score += 2

        filename = path.name.lower()

        if "carbon" in filename:
            score += 4

        if "mapper" in filename:
            score += 4

        if "observation" in filename:
            score += 2

        if score > 0:
            candidates.append({
                "path": path,
                "score": score,
                "columns": columns,
            })

    candidates.sort(
        key=lambda item: (-item["score"], str(item["path"]))
    )

    if not candidates:
        return None, []

    return candidates[0]["path"], candidates


def normalize_s2(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame(index=source.index)

    out["sample_id"] = clean_text(
        get_series(
            source,
            [
                "sample_id",
                "record_id",
                "patch_id",
                "scene_id",
                "event_id",
                "id",
            ],
        )
    )

    missing_id = out["sample_id"].isna() | (out["sample_id"] == "")

    out.loc[missing_id, "sample_id"] = [
        f"S2_ROW_{index:04d}"
        for index in source.index[missing_id]
    ]

    out["dataset_group"] = "five_site_s2_exploratory"
    out["sensor"] = "Sentinel-2"

    out["site_id"] = clean_text(
        get_series(
            source,
            [
                "site_id",
                "site",
                "site_name",
                "location",
                "source_site",
            ],
        )
    )

    out["acquisition_time_utc"] = normalize_time(
        get_series(
            source,
            [
                "acquisition_time_utc",
                "acquisition_datetime",
                "datetime_utc",
                "timestamp_utc",
                "scene_time",
                "date",
            ],
        )
    )

    out["source_latitude"] = pd.to_numeric(
        get_series(
            source,
            [
                "source_latitude",
                "latitude",
                "lat",
            ],
        ),
        errors="coerce",
    )

    out["source_longitude"] = pd.to_numeric(
        get_series(
            source,
            [
                "source_longitude",
                "longitude",
                "lon",
                "lng",
            ],
        ),
        errors="coerce",
    )

    # This five-site S2 label is not assumed to be exact physical ON/OFF ground truth.
    out["physical_release_gt"] = pd.NA
    out["metered_release_rate_kg_hr"] = pd.to_numeric(
        get_series(
            source,
            [
                "metered_release_rate_kg_hr",
                "ground_truth_rate_kg_hr",
                "release_rate_kg_hr",
                "emission_rate_kg_hr",
            ],
        ),
        errors="coerce",
    )

    out["plume_detection_label"] = pd.to_numeric(
        get_series(
            source,
            [
                "plume_detection_label",
                "label",
                "target",
                "class",
            ],
        ),
        errors="coerce",
    ).astype("Int64")

    provenance = clean_text(
        get_series(
            source,
            [
                "ground_truth_provenance",
                "provenance",
                "source_dataset",
                "dataset_source",
                "source_type",
            ],
            default="existing_five_site_s2_label",
        )
    )

    out["ground_truth_provenance"] = provenance.fillna(
        "existing_five_site_s2_label"
    )

    out["label_definition"] = (
        "exploratory plume/reference label; "
        "not uniformly verified physical source ON/OFF"
    )

    out["classification_outcome"] = pd.NA

    out["image_path"] = clean_text(
        get_series(
            source,
            [
                "image_path",
                "patch_path",
                "tif_path",
                "local_path",
                "file_path",
                "s2_path",
            ],
        )
    )

    out["original_source_file"] = str(path)
    out["original_row_index"] = source.index

    return out


def normalize_methaneair(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame(index=source.index)

    out["sample_id"] = clean_text(
        get_series(source, ["record_id", "sample_id"])
    )
    out["dataset_group"] = "methaneair_metered_controlled_release"
    out["sensor"] = "MethaneAIR"

    out["site_id"] = clean_text(
        get_series(
            source,
            ["site_region", "site_id", "flight_id"],
        )
    )

    out["acquisition_time_utc"] = normalize_time(
        get_series(
            source,
            [
                "timestamp_utc",
                "ground_truth_timestamp_utc",
                "nearest_nc_time_utc",
            ],
        )
    )

    out["source_latitude"] = pd.to_numeric(
        get_series(source, ["source_latitude"]),
        errors="coerce",
    )

    out["source_longitude"] = pd.to_numeric(
        get_series(source, ["source_longitude"]),
        errors="coerce",
    )

    out["physical_release_gt"] = pd.to_numeric(
        get_series(source, ["physical_release_gt"]),
        errors="coerce",
    ).astype("Int64")

    out["metered_release_rate_kg_hr"] = pd.to_numeric(
        get_series(
            source,
            ["metered_release_rate_kg_hr"],
        ),
        errors="coerce",
    )

    out["plume_detection_label"] = pd.NA

    out["ground_truth_provenance"] = clean_text(
        get_series(
            source,
            ["ground_truth_basis"],
            default="Stanford metered controlled-release rate",
        )
    ).fillna("Stanford metered controlled-release rate")

    out["label_definition"] = (
        "physical source ON/OFF derived from Stanford metered release rate"
    )

    out["classification_outcome"] = clean_text(
        get_series(
            source,
            ["ground_truth_status", "mapping_status"],
        )
    )

    out["image_path"] = clean_text(
        get_series(source, ["nc_path"])
    )

    out["original_source_file"] = str(path)
    out["original_row_index"] = source.index

    return out


def normalize_carbon_mapper(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame(index=source.index)

    out["sample_id"] = clean_text(
        get_series(
            source,
            [
                "observation_id",
                "sample_id",
                "record_id",
                "event_id",
                "source_id",
                "id",
            ],
        )
    )

    missing_id = out["sample_id"].isna() | (out["sample_id"] == "")

    out.loc[missing_id, "sample_id"] = [
        f"CM_ROW_{index:04d}"
        for index in source.index[missing_id]
    ]

    out["dataset_group"] = "carbon_mapper_observation_level"
    out["sensor"] = clean_text(
        get_series(
            source,
            ["sensor", "platform", "satellite"],
            default="Carbon Mapper airborne",
        )
    ).fillna("Carbon Mapper airborne")

    out["site_id"] = clean_text(
        get_series(
            source,
            [
                "site_id",
                "site",
                "site_name",
                "source_name",
                "location",
            ],
        )
    )

    out["acquisition_time_utc"] = normalize_time(
        get_series(
            source,
            [
                "acquisition_time_utc",
                "timestamp_utc",
                "datetime_utc",
                "acquisition_time",
                "date",
            ],
        )
    )

    out["source_latitude"] = pd.to_numeric(
        get_series(
            source,
            [
                "source_latitude",
                "latitude",
                "lat",
            ],
        ),
        errors="coerce",
    )

    out["source_longitude"] = pd.to_numeric(
        get_series(
            source,
            [
                "source_longitude",
                "longitude",
                "lon",
                "lng",
            ],
        ),
        errors="coerce",
    )

    # Do not infer physical ON/OFF from TP/FN/TN/FP until the file definition is verified.
    out["physical_release_gt"] = pd.NA

    out["metered_release_rate_kg_hr"] = pd.to_numeric(
        get_series(
            source,
            [
                "ground_truth_rate_kg_hr",
                "metered_release_rate_kg_hr",
                "release_rate_kg_hr",
                "emission_rate_kg_hr",
            ],
        ),
        errors="coerce",
    )

    out["plume_detection_label"] = pd.to_numeric(
        get_series(
            source,
            [
                "detection",
                "plume_detection_label",
                "label",
            ],
        ),
        errors="coerce",
    ).astype("Int64")

    out["ground_truth_provenance"] = (
        "Carbon Mapper controlled-release classification product"
    )

    out["label_definition"] = (
        "Carbon Mapper detection/classification outcome; "
        "physical ON/OFF is not inferred automatically"
    )

    out["classification_outcome"] = clean_text(
        get_series(
            source,
            [
                "tc_Classification",
                "tc_classification",
                "classification",
                "outcome",
            ],
        )
    )

    out["image_path"] = clean_text(
        get_series(
            source,
            [
                "image_path",
                "scene_path",
                "file_path",
                "plume_path",
            ],
        )
    )

    out["original_source_file"] = str(path)
    out["original_row_index"] = source.index

    return out


def main() -> None:
    if not S2_PATH.exists():
        raise FileNotFoundError(f"Missing S2 table: {S2_PATH}")

    if not METHANEAIR_PATH.exists():
        raise FileNotFoundError(
            f"Missing MethaneAIR table: {METHANEAIR_PATH}"
        )

    carbon_mapper_path = (
        OUTPUTS
        / "498_carbonmapper_observation_manifest_locked_v1.csv"
    )

    if not carbon_mapper_path.exists():
        raise FileNotFoundError(
            f"Missing locked Carbon Mapper table: {carbon_mapper_path}"
        )

    carbon_candidates = []

    frames = [
        normalize_s2(S2_PATH),
        normalize_methaneair(METHANEAIR_PATH),
    ]

    carbon_status = "not_found"

    if carbon_mapper_path is not None:
        frames.append(normalize_carbon_mapper(carbon_mapper_path))
        carbon_status = str(carbon_mapper_path)

    unified = pd.concat(frames, ignore_index=True, sort=False)

    columns = [
        "sample_id",
        "dataset_group",
        "sensor",
        "site_id",
        "acquisition_time_utc",
        "source_latitude",
        "source_longitude",
        "physical_release_gt",
        "metered_release_rate_kg_hr",
        "plume_detection_label",
        "classification_outcome",
        "label_definition",
        "ground_truth_provenance",
        "image_path",
        "original_source_file",
        "original_row_index",
    ]

    unified = unified[columns]

    duplicate_sample_id = unified["sample_id"].duplicated(
        keep=False
    )

    unified["sample_id_is_duplicate"] = duplicate_sample_id

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    unified.to_csv(OUT_CSV, index=False)

    audit_lines = [
        "Unified methane provenance table audit",
        "=" * 70,
        f"Output rows: {len(unified)}",
        f"S2 source: {S2_PATH}",
        f"MethaneAIR source: {METHANEAIR_PATH}",
        f"Carbon Mapper source: {carbon_status}",
        "",
        "Rows by dataset_group:",
        unified["dataset_group"].value_counts(dropna=False).to_string(),
        "",
        "Rows by sensor:",
        unified["sensor"].value_counts(dropna=False).to_string(),
        "",
        "Physical release ground truth availability:",
        unified["physical_release_gt"].notna()
        .value_counts()
        .rename(index={True: "available", False: "missing"})
        .to_string(),
        "",
        "Metered release-rate availability:",
        unified["metered_release_rate_kg_hr"].notna()
        .value_counts()
        .rename(index={True: "available", False: "missing"})
        .to_string(),
        "",
        "Duplicate sample IDs:",
        str(int(duplicate_sample_id.sum())),
        "",
        "Carbon Mapper candidates:",
    ]

    if carbon_candidates:
        for item in carbon_candidates[:10]:
            audit_lines.append(
                f"score={item['score']:2d}  {item['path']}"
            )
    else:
        audit_lines.append("NONE")

    audit_lines.extend([
        "",
        "Important interpretation:",
        "- MethaneAIR rows contain verified physical ON/OFF ground truth.",
        "- Five-site Sentinel-2 rows retain exploratory plume/reference labels.",
        "- Carbon Mapper classification is kept separate and is not automatically converted to physical ON/OFF.",
    ])

    OUT_AUDIT.write_text(
        "\n".join(audit_lines),
        encoding="utf-8",
    )

    print(f"Created: {OUT_CSV}")
    print(f"Created: {OUT_AUDIT}")
    print()
    print(unified["dataset_group"].value_counts(dropna=False))
    print()
    print("Carbon Mapper source:", carbon_status)


if __name__ == "__main__":
    main()
