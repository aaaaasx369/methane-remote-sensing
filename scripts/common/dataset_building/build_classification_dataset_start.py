import pandas as pd
from pathlib import Path
import json

INPUT = Path("outputs/final_controlled_release_satellite_availability.csv")

SUMMARY_OUT = Path("outputs/11_availability_summary.csv")
CANDIDATE_OUT = Path("outputs/12_dataset_candidate_events.csv")
LABEL_OUT = Path("outputs/12_classification_labels.csv")
GEE_JS_OUT = Path("outputs/13_export_s2_landsat_patches.js")

df = pd.read_csv(INPUT, low_memory=False)

# -----------------------------
# Clean numeric columns
# -----------------------------
count_cols = ["s2_count", "l8_count", "l9_count", "emit_l2a_count"]
for col in count_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    else:
        df[col] = 0

cloud_cols = ["s2_first_cloud", "l8_first_cloud", "l9_first_cloud"]
for col in cloud_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        df[col] = pd.NA

for col in ["emission_tph_mean", "emission_tph_median", "emission_tph_max"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# -----------------------------
# Make classification label
# label = 1 means methane release
# label = 0 means no release
# -----------------------------
if "true_release" in df.columns:
    df["label"] = pd.to_numeric(df["true_release"], errors="coerce")
else:
    df["label"] = (df["emission_tph_max"].fillna(0) > 0).astype(int)

# 如果 true_release 是空，就用 emission_tph_max 補
missing_label = df["label"].isna()
df.loc[missing_label, "label"] = (df.loc[missing_label, "emission_tph_max"].fillna(0) > 0).astype(int)

df["label"] = df["label"].astype(int)

# -----------------------------
# Define usable public satellite observations
# 先不要太嚴格，雲量門檻先設 40%
# -----------------------------
S2_CLOUD_THRESHOLD = 40
LANDSAT_CLOUD_THRESHOLD = 40

df["usable_s2"] = (
    (df["s2_count"] > 0)
    & (
        df["s2_first_cloud"].isna()
        | (df["s2_first_cloud"] <= S2_CLOUD_THRESHOLD)
    )
)

df["usable_landsat8"] = (
    (df["l8_count"] > 0)
    & (
        df["l8_first_cloud"].isna()
        | (df["l8_first_cloud"] <= LANDSAT_CLOUD_THRESHOLD)
    )
)

df["usable_landsat9"] = (
    (df["l9_count"] > 0)
    & (
        df["l9_first_cloud"].isna()
        | (df["l9_first_cloud"] <= LANDSAT_CLOUD_THRESHOLD)
    )
)

df["usable_landsat"] = df["usable_landsat8"] | df["usable_landsat9"]

df["usable_emit"] = df["emit_l2a_count"] > 0

df["usable_any_public_image"] = (
    df["usable_s2"]
    | df["usable_landsat"]
    | df["usable_emit"]
)

# -----------------------------
# Summary table
# -----------------------------
summary = pd.DataFrame({
    "item": [
        "total_events",
        "positive_release_events",
        "negative_no_release_events",
        "events_with_sentinel2",
        "events_with_usable_sentinel2_cloud_filtered",
        "events_with_landsat8",
        "events_with_landsat9",
        "events_with_usable_landsat",
        "events_with_emit_l2a",
        "events_with_any_public_image",
    ],
    "count": [
        len(df),
        int((df["label"] == 1).sum()),
        int((df["label"] == 0).sum()),
        int((df["s2_count"] > 0).sum()),
        int(df["usable_s2"].sum()),
        int((df["l8_count"] > 0).sum()),
        int((df["l9_count"] > 0).sum()),
        int(df["usable_landsat"].sum()),
        int((df["emit_l2a_count"] > 0).sum()),
        int(df["usable_any_public_image"].sum()),
    ]
})

summary.to_csv(SUMMARY_OUT, index=False)

# -----------------------------
# Candidate events for dataset
# 先以 Sentinel-2 + Landsat 為主，EMIT 之後再處理
# -----------------------------
candidate = df[df["usable_s2"] | df["usable_landsat"]].copy()

candidate.to_csv(CANDIDATE_OUT, index=False)

label_cols = [
    "event_id",
    "paper",
    "date_utc",
    "datetime_utc",
    "lat",
    "lon",
    "label",
    "emission_tph_mean",
    "emission_tph_median",
    "emission_tph_max",
    "usable_s2",
    "usable_landsat8",
    "usable_landsat9",
    "usable_emit",
]

for c in label_cols:
    if c not in candidate.columns:
        candidate[c] = ""

candidate[label_cols].to_csv(LABEL_OUT, index=False)

print("Saved:", SUMMARY_OUT)
print("Saved:", CANDIDATE_OUT)
print("Saved:", LABEL_OUT)

print("\nSummary:")
print(summary.to_string(index=False))

print("\nCandidate events:", len(candidate))
print("\nLabel balance:")
print(candidate["label"].value_counts())

# -----------------------------
# Generate GEE JavaScript for exporting image patches
# -----------------------------
# 先測試少量，不要一次 export 全部
MAX_EVENTS_TO_EXPORT = 10

export_df = candidate.head(MAX_EVENTS_TO_EXPORT).copy()

features = []

for _, r in export_df.iterrows():
    lon = float(r["lon"])
    lat = float(r["lat"])

    props = {
        "event_id": str(r.get("event_id", "")),
        "paper": str(r.get("paper", "")),
        "datetime_utc": str(r.get("datetime_utc", "")),
        "date_utc": str(r.get("date_utc", "")),
        "label": str(r.get("label", "")),
        "emission_tph_max": "" if pd.isna(r.get("emission_tph_max", pd.NA)) else str(r.get("emission_tph_max")),
        "usable_s2": str(r.get("usable_s2", "")),
        "usable_landsat8": str(r.get("usable_landsat8", "")),
        "usable_landsat9": str(r.get("usable_landsat9", "")),
    }

    features.append(
        f"  ee.Feature(ee.Geometry.Point([{lon}, {lat}]), {json.dumps(props)})"
    )

features_joined = ",\n".join(features)

gee_code = f"""
// Auto-generated by build_classification_dataset_start.py
// Purpose:
// Export Sentinel-2 and Landsat image patches for methane classification dataset.
// Start with 10 events only. After confirming it works, increase MAX_EVENTS_TO_EXPORT in Python.

var events = ee.FeatureCollection([
{features_joined}
]);

Map.centerObject(events, 8);
Map.addLayer(events, {{color: 'red'}}, 'candidate methane events');

var S2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED');
var L8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2');
var L9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2');

var WINDOW_HOURS = 24;
var PATCH_RADIUS_METERS = 1000;

// Output Google Drive folder
var DRIVE_FOLDER = 'methane_image_patches';

// Sentinel-2 bands useful for methane / RGB / NIR / SWIR
var S2_BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'];

// Landsat SR bands: blue, green, red, NIR, SWIR1, SWIR2
var LANDSAT_BANDS = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'];

function cleanName(s) {{
  return ee.String(s)
    .replace(':', '_')
    .replace('-', '_')
    .replace('-', '_')
    .replace('T', '_')
    .replace('Z', '');
}}

var eventList = events.toList(events.size());
var n = events.size().getInfo();

print('Number of events to export', n);

for (var i = 0; i < n; i++) {{
  var event = ee.Feature(eventList.get(i));

  var point = event.geometry();
  var region = point.buffer(PATCH_RADIUS_METERS).bounds();

  var t = ee.Date(event.get('datetime_utc'));
  var start = t.advance(-WINDOW_HOURS, 'hour');
  var end = t.advance(WINDOW_HOURS, 'hour');

  var eventId = event.get('event_id').getInfo();
  var label = event.get('label').getInfo();

  // -----------------------------
  // Export Sentinel-2 patch
  // -----------------------------
  var s2 = S2
    .filterBounds(point)
    .filterDate(start, end)
    .sort('CLOUDY_PIXEL_PERCENTAGE');

  var s2Count = s2.size().getInfo();

  if (s2Count > 0) {{
    var s2Img = ee.Image(s2.first()).select(S2_BANDS);

    Export.image.toDrive({{
      image: s2Img,
      description: 'S2_' + eventId + '_label_' + label,
      folder: DRIVE_FOLDER,
      fileNamePrefix: 'S2_' + eventId + '_label_' + label,
      region: region,
      scale: 20,
      maxPixels: 1e9
    }});
  }}

  // -----------------------------
  // Export Landsat 8/9 patch
  // -----------------------------
  var landsat = L8.merge(L9)
    .filterBounds(point)
    .filterDate(start, end)
    .sort('CLOUD_COVER');

  var landsatCount = landsat.size().getInfo();

  if (landsatCount > 0) {{
    var lsImg = ee.Image(landsat.first()).select(LANDSAT_BANDS);

    Export.image.toDrive({{
      image: lsImg,
      description: 'Landsat_' + eventId + '_label_' + label,
      folder: DRIVE_FOLDER,
      fileNamePrefix: 'Landsat_' + eventId + '_label_' + label,
      region: region,
      scale: 30,
      maxPixels: 1e9
    }});
  }}
}}
"""

GEE_JS_OUT.write_text(gee_code)

print("Saved:", GEE_JS_OUT)
print("\nNext:")
print("1. Open outputs/13_export_s2_landsat_patches.js")
print("2. Paste it into Google Earth Engine Code Editor")
print("3. Run it")
print("4. In Tasks, run the export tasks")