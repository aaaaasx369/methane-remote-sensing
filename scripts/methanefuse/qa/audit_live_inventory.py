from pathlib import Path
from collections import Counter
import json
import pandas as pd

ROOT = Path("/project/6002520/yunjung1/MethaneFuse")
OUT = ROOT / "outputs/live_inventory_audit"
OUT.mkdir(parents=True, exist_ok=True)

SCAN_DIRS = [
    ROOT / "data",
    ROOT / "outputs",
]

LABELS = [
    "label",
    "ground_truth_label",
    "binary_label",
    "target",
]

IDS = [
    "record_id",
    "event_id",
    "plume_id",
    "sample_id",
    "observation_id",
    "acquisition_id",
    "dedup_key",
]

SENSORS = [
    "sensor",
    "satellite",
    "platform",
    "landsat_sensor",
]

SOURCES = [
    "ground_truth_source",
    "dataset_source",
    "data_source",
    "source_type",
]

TIMES = [
    "event_time_utc",
    "datetime_utc",
    "acquisition_time_utc",
    "time_coverage_start",
    "timestamp_utc",
]


def choose(columns, candidates):
    lookup = {
        str(column).lower(): column
        for column in columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def counts_text(series):
    counts = (
        series
        .fillna("<missing>")
        .astype(str)
        .value_counts(dropna=False)
    )

    return "; ".join(
        f"{name}:{count}"
        for name, count in counts.head(30).items()
    )


table_rows = []
status_rows = []

files = []

for scan_dir in SCAN_DIRS:
    if not scan_dir.exists():
        continue

    files.extend(scan_dir.rglob("*.csv"))
    files.extend(scan_dir.rglob("*.tsv"))
    files.extend(scan_dir.rglob("*.parquet"))

files = sorted(set(files))

for number, path in enumerate(files, start=1):
    relative = str(path.relative_to(ROOT))

    try:
        if path.suffix == ".csv":
            df = pd.read_csv(
                path,
                low_memory=False,
                on_bad_lines="warn",
            )
        elif path.suffix == ".tsv":
            df = pd.read_csv(
                path,
                sep="\t",
                low_memory=False,
                on_bad_lines="warn",
            )
        else:
            df = pd.read_parquet(path)

        columns = list(df.columns)

        label_col = choose(columns, LABELS)
        id_col = choose(columns, IDS)
        sensor_col = choose(columns, SENSORS)
        source_col = choose(columns, SOURCES)
        time_col = choose(columns, TIMES)

        row = {
            "relative_path": relative,
            "size_mb": round(
                path.stat().st_size / 1024 / 1024,
                3,
            ),
            "rows": len(df),
            "columns": len(columns),
            "id_column": id_col or "",
            "unique_ids": (
                df[id_col].nunique(dropna=True)
                if id_col
                else ""
            ),
            "label_column": label_col or "",
            "label_counts": (
                counts_text(df[label_col])
                if label_col
                else ""
            ),
            "sensor_column": sensor_col or "",
            "sensor_counts": (
                counts_text(df[sensor_col])
                if sensor_col
                else ""
            ),
            "source_column": source_col or "",
            "source_counts": (
                counts_text(df[source_col])
                if source_col
                else ""
            ),
            "time_column": time_col or "",
            "time_min": "",
            "time_max": "",
            "parse_error": "",
        }

        if time_col:
            parsed = pd.to_datetime(
                df[time_col],
                errors="coerce",
                utc=True,
            ).dropna()

            if not parsed.empty:
                row["time_min"] = parsed.min().isoformat()
                row["time_max"] = parsed.max().isoformat()

        table_rows.append(row)

        for column in columns:
            name = str(column)
            low = name.lower()

            relevant = (
                low == "status"
                or low.endswith("_status")
                or low.endswith("_qa_pass")
                or low.endswith("_downloaded")
                or low in {
                    "qa_pass",
                    "model_ready",
                    "qa_model_ready",
                    "strict_model_ready",
                    "label_status",
                }
            )

            if relevant:
                status_rows.append({
                    "relative_path": relative,
                    "column": name,
                    "value_counts": counts_text(df[name]),
                })

    except Exception as error:
        table_rows.append({
            "relative_path": relative,
            "size_mb": round(
                path.stat().st_size / 1024 / 1024,
                3,
            ),
            "rows": "",
            "columns": "",
            "id_column": "",
            "unique_ids": "",
            "label_column": "",
            "label_counts": "",
            "sensor_column": "",
            "sensor_counts": "",
            "source_column": "",
            "source_counts": "",
            "time_column": "",
            "time_min": "",
            "time_max": "",
            "parse_error": (
                f"{type(error).__name__}: {error}"
            ),
        })

    if number % 50 == 0:
        print(
            f"Processed {number}/{len(files)} tables"
        )


table_df = pd.DataFrame(table_rows)
status_df = pd.DataFrame(status_rows)

table_path = OUT / "live_tabular_inventory.csv"
status_path = OUT / "live_status_qa_inventory.csv"

table_df.to_csv(table_path, index=False)
status_df.to_csv(status_path, index=False)

manifest_path = (
    ROOT
    / "data/methaneair_full/"
    / "sentinel2_temporal_manifest.csv"
)

summary = []

if manifest_path.exists():
    manifest = pd.read_csv(
        manifest_path,
        low_memory=False,
    )

    summary.append({
        "metric": "records",
        "value": len(manifest),
    })

    for column in [
        "label_status",
        "t0_status",
        "t90_status",
        "t360_status",
    ]:
        if column in manifest.columns:
            for value, count in (
                manifest[column]
                .fillna("<missing>")
                .astype(str)
                .value_counts()
                .items()
            ):
                summary.append({
                    "metric": f"{column}:{value}",
                    "value": int(count),
                })

    for column in [
        "t0_qa_pass",
        "t90_qa_pass",
        "t360_qa_pass",
        "all_three_downloaded",
        "all_three_qa_pass",
    ]:
        if column in manifest.columns:
            values = (
                manifest[column]
                .fillna(False)
                .astype(str)
                .str.lower()
                .isin(["true", "1", "yes"])
            )

            summary.append({
                "metric": column,
                "value": int(values.sum()),
            })

    for prefix in ["t0", "t90", "t360"]:
        status_col = f"{prefix}_status"
        scene_col = f"{prefix}_scene_id"
        time_col = f"{prefix}_scene_time_utc"

        if status_col in manifest.columns:
            already = (
                manifest[status_col]
                .astype(str)
                .eq("already_exists")
            )

            if scene_col in manifest.columns:
                summary.append({
                    "metric": (
                        f"{prefix}_already_exists_"
                        "missing_scene_id"
                    ),
                    "value": int(
                        (
                            already
                            & manifest[scene_col].isna()
                        ).sum()
                    ),
                })

            if time_col in manifest.columns:
                summary.append({
                    "metric": (
                        f"{prefix}_already_exists_"
                        "missing_scene_time"
                    ),
                    "value": int(
                        (
                            already
                            & manifest[time_col].isna()
                        ).sum()
                    ),
                })


summary_df = pd.DataFrame(summary)

summary_path = (
    OUT
    / "sentinel2_temporal_summary.csv"
)

summary_df.to_csv(summary_path, index=False)

print()
print("Audit complete")
print("Tables:", len(table_df))
print("Output:", OUT)
print()
print("Sentinel-2 temporal summary:")
print(summary_df.to_string(index=False))
