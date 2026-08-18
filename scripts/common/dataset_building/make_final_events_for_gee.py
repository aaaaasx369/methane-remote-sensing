import pandas as pd
from pathlib import Path
import json

INPUT = Path("outputs/07_unique_overpass_events.csv")
OUTPUT = Path("outputs/10_final_events_for_gee.csv")
DROPPED = Path("outputs/10_dropped_unknown_non_satellite.csv")
GEE_JS = Path("outputs/10_gee_check_final_events.js")

df = pd.read_csv(INPUT, low_memory=False)

# -----------------------------
# Clean satellite names
# -----------------------------
def clean_satellite_name(x):
    s = str(x).strip()

    mapping = {
        "WorldView 3": "WorldView-3",
        "WV3": "WorldView-3",
        "LandSat": "Landsat",
        "Landsat 8": "Landsat",
        "Landsat 9": "Landsat",
        "GHGSat CX": "GHGSat",
        "GHGSat C2": "GHGSat",
        "EnMap": "EnMAP",
        "GF5": "Gaofen-5",
        "ZY1": "Ziyuan-1",
        "HJ2B": "Huanjing-2",
    }

    return mapping.get(s, s)

df["satellite_clean"] = df["satellite_from_paper"].apply(clean_satellite_name)

# -----------------------------
# Decide which Unknown rows to keep/drop
# -----------------------------
df["source_files_str"] = df["source_files"].astype(str)

bad_unknown_keywords = [
    "SciAv",
    "SOOFIE",
    "Bridger",
    "MAIR",
    "MethaneAIR",
    "00_raw_reports",
]

good_unknown_keywords = [
    "matchedDF_Satellites",
]

def classify_row(row):
    sat = str(row["satellite_clean"]).strip()
    src = str(row["source_files_str"])

    if sat != "Unknown":
        return "keep_known_satellite"

    # Unknown but source looks like satellite matched dataframe
    if any(k in src for k in good_unknown_keywords) and not any(k in src for k in bad_unknown_keywords):
        return "keep_unknown_satellite_matched_review"

    # Unknown and source looks like aircraft/raw report
    return "drop_unknown_non_satellite_or_raw"

df["row_status"] = df.apply(classify_row, axis=1)

final = df[df["row_status"].isin([
    "keep_known_satellite",
    "keep_unknown_satellite_matched_review",
])].copy()

dropped = df[df["row_status"] == "drop_unknown_non_satellite_or_raw"].copy()

# -----------------------------
# Save outputs
# -----------------------------
final.to_csv(OUTPUT, index=False)
dropped.to_csv(DROPPED, index=False)

print("Saved final table:", OUTPUT)
print("Saved dropped rows:", DROPPED)

print("\nOriginal rows:", len(df))
print("Final rows:", len(final))
print("Dropped rows:", len(dropped))

print("\nCounts by paper:")
print(final["paper"].value_counts())

print("\nCounts by satellite_clean:")
print(final["satellite_clean"].value_counts())

print("\nRow status counts:")
print(df["row_status"].value_counts())

# -----------------------------
# Generate GEE JS
# -----------------------------
features = []

for _, r in final.iterrows():
    lon = float(r["lon"])
    lat = float(r["lat"])

    props = {
        "paper": str(r["paper"]),
        "event_id": str(r["event_id"]),
        "datetime_utc": str(r["datetime_utc"]),
        "date_utc": str(r["date_utc"]),
        "satellite_from_paper": str(r["satellite_clean"]),
        "true_release": str(r["true_release"]),
        "emission_tph_mean": "" if pd.isna(r["emission_tph_mean"]) else str(r["emission_tph_mean"]),
        "emission_tph_median": "" if pd.isna(r["emission_tph_median"]) else str(r["emission_tph_median"]),
        "emission_tph_max": "" if pd.isna(r["emission_tph_max"]) else str(r["emission_tph_max"]),
        "row_status": str(r["row_status"]),
    }

    features.append(
        f"  ee.Feature(ee.Geometry.Point([{lon}, {lat}]), {json.dumps(props)})"
    )

features_joined = ",\n".join(features)

gee_code = f"""
// Auto-generated from outputs/10_final_events_for_gee.csv
// Paste into Google Earth Engine Code Editor.

var events = ee.FeatureCollection([
{features_joined}
]);

Map.centerObject(events, 8);
Map.addLayer(events, {{color: 'red'}}, 'final methane controlled release events');

var S2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED');
var L8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2');
var L9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2');
var EMIT = ee.ImageCollection('NASA/EMIT/L2A/RFL');

// 初步查 availability 用 ±24 小時。
// 如果老師要更嚴格，可以改成 6 或 3。
var WINDOW_HOURS = 24;

function firstTimeOrNone(collection) {{
  return ee.Algorithms.If(
    collection.size().gt(0),
    ee.Date(collection.first().get('system:time_start')).format('YYYY-MM-dd HH:mm:ss'),
    'none'
  );
}}

function checkAvailability(feature) {{
  var point = feature.geometry();
  var t = ee.Date(feature.get('datetime_utc'));

  var start = t.advance(-WINDOW_HOURS, 'hour');
  var end = t.advance(WINDOW_HOURS, 'hour');

  var s2 = S2
    .filterBounds(point)
    .filterDate(start, end)
    .sort('CLOUDY_PIXEL_PERCENTAGE');

  var l8 = L8
    .filterBounds(point)
    .filterDate(start, end)
    .sort('CLOUD_COVER');

  var l9 = L9
    .filterBounds(point)
    .filterDate(start, end)
    .sort('CLOUD_COVER');

  var emit = EMIT
    .filterBounds(point)
    .filterDate(start, end);

  return feature.set({{
    search_start: start.format('YYYY-MM-dd HH:mm:ss'),
    search_end: end.format('YYYY-MM-dd HH:mm:ss'),

    s2_count: s2.size(),
    l8_count: l8.size(),
    l9_count: l9.size(),
    emit_l2a_count: emit.size(),

    s2_first_time: firstTimeOrNone(s2),
    l8_first_time: firstTimeOrNone(l8),
    l9_first_time: firstTimeOrNone(l9),
    emit_l2a_first_time: firstTimeOrNone(emit),

    s2_first_cloud: ee.Algorithms.If(
      s2.size().gt(0),
      ee.Image(s2.first()).get('CLOUDY_PIXEL_PERCENTAGE'),
      'none'
    ),

    l8_first_cloud: ee.Algorithms.If(
      l8.size().gt(0),
      ee.Image(l8.first()).get('CLOUD_COVER'),
      'none'
    ),

    l9_first_cloud: ee.Algorithms.If(
      l9.size().gt(0),
      ee.Image(l9.first()).get('CLOUD_COVER'),
      'none'
    )
  }});
}}

var checked = events.map(checkAvailability);

print('Checked availability table', checked);
print('Number of final events', checked.size());

Export.table.toDrive({{
  collection: checked,
  description: 'final_controlled_release_satellite_availability',
  fileFormat: 'CSV'
}});
"""

GEE_JS.write_text(gee_code)

print("\nSaved GEE JS:", GEE_JS)