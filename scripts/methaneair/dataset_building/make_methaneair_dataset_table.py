from pathlib import Path
import re
import pandas as pd


# =========================
# Paths
# =========================
patch_dir = Path("sample_patches/methaneair_s2")
index_path = Path("outputs/16_methaneair_s2_patch_index.csv")
quality_path = Path("outputs/17_methaneair_s2_patch_quality.csv")
out_path = Path("outputs/18_methaneair_s2_dataset_table.csv")


# =========================
# Helper functions
# =========================
def safe_get(row, col, default=""):
    """
    Safely get value from pandas Series.
    If the column does not exist or the value is NaN, return default.
    """
    if col not in row.index:
        return default

    value = row[col]

    if pd.isna(value):
        return default

    return value


def clean_id_value(value):
    """
    Convert ID-like values into clean strings.
    Example:
    8.0 -> "8"
    "MX024" -> "MX024"
    """
    if value == "":
        return ""

    try:
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
    except Exception:
        pass

    return str(value)


# =========================
# Load metadata tables
# =========================
if not index_path.exists():
    raise FileNotFoundError(f"Cannot find index file: {index_path}")

if not patch_dir.exists():
    raise FileNotFoundError(f"Cannot find patch folder: {patch_dir}")

index_df = pd.read_csv(index_path)

# Make sure export_index is numeric
if "export_index" not in index_df.columns:
    raise ValueError("Missing column 'export_index' in patch index file.")

index_df["export_index"] = pd.to_numeric(index_df["export_index"], errors="coerce")


# =========================
# Build dataset table
# =========================
rows = []

for tif_path in sorted(patch_dir.glob("*.tif")):
    # Extract index from filename, for example:
    # MA_S2_patch_0.tif -> 0
    # MA_S2_patch_09.tif -> 9
    match = re.search(r"MA_S2_patch_(\d+)", tif_path.stem)

    if match is None:
        print("Skip unknown filename:", tif_path.name)
        continue

    export_index = int(match.group(1))

    # Find matching event metadata from GEE patch index
    matched_rows = index_df[index_df["export_index"] == export_index]

    if len(matched_rows) == 0:
        print("No matching index for:", tif_path.name)
        continue

    event_row = matched_rows.iloc[0]

    flight_id = clean_id_value(safe_get(event_row, "flight_id", ""))
    plume_id = clean_id_value(safe_get(event_row, "plume_id", ""))

    # Create event_id using flight_id + plume_id
    # Example: MethaneAIR_MX024_8
    event_id = f"MethaneAIR_{flight_id}_{plume_id}"

    rows.append({
        # Image information
        "filename": tif_path.name,
        "relative_path": str(tif_path),
        "export_index": export_index,

        # Classification label
        "label": 1,
        "label_type": "detected_methane",
        "source_dataset": "MethaneAIR_L4point",
        "ground_truth_type": "observational_detection",
        "sensor": "Sentinel-2",

        # Event metadata
        "event_id": event_id,
        "datetime_utc": safe_get(event_row, "datetime_utc", ""),
        "lat": safe_get(event_row, "lat", ""),
        "lon": safe_get(event_row, "lon", ""),
        "emission_kg_hr": safe_get(
            event_row,
            "emission_kg_hr",
            safe_get(event_row, "flux", "")
        ),
        "emission_tph": safe_get(event_row, "emission_tph", ""),
        "flight_id": flight_id,
        "plume_id": plume_id
    })


dataset_df = pd.DataFrame(rows)


# =========================
# Merge patch quality table
# =========================
if quality_path.exists():
    quality_df = pd.read_csv(quality_path)

    if "filename" in quality_df.columns:
        dataset_df = dataset_df.merge(
            quality_df,
            on="filename",
            how="left"
        )
    else:
        print("Warning: quality file exists but has no 'filename' column.")
else:
    print("Warning: quality file not found:", quality_path)


# =========================
# Save output
# =========================
out_path.parent.mkdir(parents=True, exist_ok=True)
dataset_df.to_csv(out_path, index=False)

print(dataset_df)
print("Saved:", out_path)
print("Number of samples:", len(dataset_df))

if "event_id" in dataset_df.columns:
    print("Event IDs:")
    print(dataset_df["event_id"])