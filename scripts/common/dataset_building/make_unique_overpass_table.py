import pandas as pd
from pathlib import Path
import json

INPUT = Path("outputs/06_strict_event_table_for_gee.csv")
OUTPUT = Path("outputs/07_unique_overpass_events.csv")
GEE_JS = Path("outputs/08_gee_check_unique_events.js")

df = pd.read_csv(INPUT, low_memory=False)

df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], errors="coerce", utc=True)
df = df.dropna(subset=["datetime_utc", "lat", "lon"])

df["emission_tph"] = pd.to_numeric(df["emission_tph"], errors="coerce")

for col in ["paper", "site_name", "satellite", "team", "source_file"]:
    if col not in df.columns:
        df[col] = ""
    df[col] = df[col].astype(str).fillna("").str.strip()

# 清掉奇怪空值
df.loc[df["satellite"].isin(["nan", "None", "NaN"]), "satellite"] = ""
df.loc[df["team"].isin(["nan", "None", "NaN"]), "team"] = ""

# 如果 satellite 空白，先用 source_file 猜
def infer_satellite(row):
    sat = str(row["satellite"]).strip()
    if sat:
        return sat

    s = str(row["source_file"]).lower()

    if "sentinel" in s or "s2" in s:
        return "Sentinel-2"
    if "landsat" in s or "ls8" in s or "ls9" in s:
        return "Landsat"
    if "prisma" in s:
        return "PRISMA"
    if "ghgsat" in s or "gsc" in s:
        return "GHGSat"
    if "worldview" in s or "wv3" in s:
        return "WorldView-3"
    if "enmap" in s:
        return "EnMAP"
    if "gaofen" in s or "gf5" in s:
        return "Gaofen-5"
    if "ziyuan" in s or "zy1" in s:
        return "Ziyuan-1"
    if "huanjing" in s or "hj2" in s:
        return "Huanjing-2"

    return "Unknown"

df["satellite_clean"] = df.apply(infer_satellite, axis=1)

# ------------------------------------------------------------
# 核心：把很接近的時間合併成同一個 overpass
# 同一 paper + site + satellite，如果時間差小於 30 分鐘，視為同一事件
# ------------------------------------------------------------

df = df.sort_values(["paper", "site_name", "satellite_clean", "datetime_utc"])

cluster_rows = []

cluster_gap_minutes = 30

for (paper, site, satellite), g in df.groupby(["paper", "site_name", "satellite_clean"]):
    g = g.sort_values("datetime_utc").copy()

    cluster_id = 0
    prev_time = None
    cluster_ids = []

    for t in g["datetime_utc"]:
        if prev_time is None:
            cluster_id += 1
        else:
            gap = (t - prev_time).total_seconds() / 60
            if gap > cluster_gap_minutes:
                cluster_id += 1

        cluster_ids.append(cluster_id)
        prev_time = t

    g["cluster_id"] = cluster_ids

    for cid, c in g.groupby("cluster_id"):
        emission_nonnull = c["emission_tph"].dropna()

        if len(emission_nonnull) > 0:
            emission_tph_mean = emission_nonnull.mean()
            emission_tph_median = emission_nonnull.median()
            emission_tph_max = emission_nonnull.max()
            true_release = int(emission_tph_max > 0)
        else:
            emission_tph_mean = None
            emission_tph_median = None
            emission_tph_max = None
            true_release = ""

        # 用中位數時間代表這個 overpass cluster
        times_int = c["datetime_utc"].astype("int64")
        median_time = pd.to_datetime(times_int.median(), utc=True)

        source_files = sorted(set(c["source_file"].astype(str)))
        teams = sorted(set([x for x in c["team"].astype(str) if x and x != "nan"]))

        cluster_rows.append({
            "paper": paper,
            "site_name": site,
            "event_id": f"{paper}_{site}_{satellite}_{cid}",
            "datetime_utc": median_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "date_utc": median_time.strftime("%Y-%m-%d"),
            "lat": c["lat"].astype(float).median(),
            "lon": c["lon"].astype(float).median(),
            "satellite_from_paper": satellite,
            "true_release": true_release,
            "emission_tph_mean": emission_tph_mean,
            "emission_tph_median": emission_tph_median,
            "emission_tph_max": emission_tph_max,
            "n_rows_merged": len(c),
            "n_unique_times_merged": c["datetime_utc"].nunique(),
            "teams": ";".join(teams),
            "source_files": ";".join(source_files[:5]),
        })

out = pd.DataFrame(cluster_rows)

# 排序
out["datetime_utc_sort"] = pd.to_datetime(out["datetime_utc"], utc=True)
out = out.sort_values(["paper", "datetime_utc_sort", "satellite_from_paper"])
out = out.drop(columns=["datetime_utc_sort"])

out.to_csv(OUTPUT, index=False)

print(f"Saved unique overpass table to: {OUTPUT}")
print(f"Rows before merge: {len(df)}")
print(f"Rows after merge: {len(out)}")

print("\nCounts by paper:")
print(out["paper"].value_counts())

print("\nCounts by satellite:")
print(out["satellite_from_paper"].value_counts())

print("\nPreview:")
print(out.head(50).to_string(index=False))


# ------------------------------------------------------------
# 產生 GEE JS：用 unique overpass events 查 S2 / Landsat / EMIT 是否存在
# ------------------------------------------------------------

features = []

for _, r in out.iterrows():
    lon = float(r["lon"])
    lat = float(r["lat"])

    props = {
        "paper": str(r["paper"]),
        "event_id": str(r["event_id"]),
        "datetime_utc": str(r["datetime_utc"]),
        "date_utc": str(r["date_utc"]),
        "satellite_from_paper": str(r["satellite_from_paper"]),
        "true_release": str(r["true_release"]),
        "emission_tph_mean": "" if pd.isna(r["emission_tph_mean"]) else str(r["emission_tph_mean"]),
        "emission_tph_median": "" if pd.isna(r["emission_tph_median"]) else str(r["emission_tph_median"]),
        "emission_tph_max": "" if pd.isna(r["emission_tph_max"]) else str(r["emission_tph_max"]),
        "n_rows_merged": str(r["n_rows_merged"]),
    }

    features.append(
        f"  ee.Feature(ee.Geometry.Point([{lon}, {lat}]), {json.dumps(props)})"
    )

features_joined = ",\n".join(features)

gee_code = f"""
// Auto-generated from 07_unique_overpass_events.csv
// Paste into Google Earth Engine Code Editor.

var events = ee.FeatureCollection([
{features_joined}
]);

Map.centerObject(events, 8);
Map.addLayer(events, {{color: 'red'}}, 'unique methane release events');

var S2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED');
var L8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2');
var L9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2');
var EMIT = ee.ImageCollection('NASA/EMIT/L2A/RFL');

// 初步 availability check 用 ±24 小時。
// 之後如果要更嚴格，可以改成 3 或 6 小時。
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
print('Number of unique events', checked.size());

Export.table.toDrive({{
  collection: checked,
  description: 'unique_controlled_release_satellite_availability',
  fileFormat: 'CSV'
}});

// Optional: show first Sentinel-2 image near first event
var firstEvent = ee.Feature(events.first());
var firstPoint = firstEvent.geometry();
var firstTime = ee.Date(firstEvent.get('datetime_utc'));

var firstS2 = ee.Image(
  S2.filterBounds(firstPoint)
    .filterDate(firstTime.advance(-WINDOW_HOURS, 'hour'), firstTime.advance(WINDOW_HOURS, 'hour'))
    .sort('CLOUDY_PIXEL_PERCENTAGE')
    .first()
);

Map.addLayer(
  firstS2,
  {{bands: ['B12', 'B11', 'B8'], min: 0, max: 4000}},
  'First S2 SWIR image',
  false
);
"""

GEE_JS.write_text(gee_code)

print(f"Saved GEE JS to: {GEE_JS}")