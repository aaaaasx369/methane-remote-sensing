from pathlib import Path
import io
import re
import time
import zipfile
import requests
import pandas as pd
import numpy as np
import ee


# =========================
# Settings
# =========================
PROJECT = "methane-release-gee"

PATCH_RADIUS = 1000
SCALE = 20
BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]

START_INDEX = 0
N_DOWNLOAD = None   # 先測試 20 張；成功後再改成 None 或 84

OUT_DIR = Path("sample_patches/controlled_release_s2")
OUT_INDEX = Path("outputs/20_controlled_release_s2_patch_index.csv")

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_INDEX.parent.mkdir(parents=True, exist_ok=True)


INPUT_CANDIDATES = [
    Path("outputs/12_dataset_candidate_events_with_latlon.csv"),
]


# =========================
# Helpers
# =========================
def initialize_earth_engine():
    try:
        ee.Initialize(project=PROJECT)
        print("Earth Engine initialized.")
    except Exception:
        print("Need Earth Engine authentication.")
        ee.Authenticate()
        ee.Initialize(project=PROJECT)
        print("Earth Engine initialized after authentication.")


def pick_col(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def clean_text(x):
    x = str(x)
    x = re.sub(r"[^A-Za-z0-9_\-]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_")


def parse_label(value):
    if pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        return int(float(value) > 0)

    s = str(value).strip().lower()

    if s in ["1", "true", "yes", "release", "released", "positive", "plume"]:
        return 1

    if s in ["0", "false", "no", "no_release", "negative", "none"]:
        return 0

    try:
        return int(float(s) > 0)
    except Exception:
        return None


def download_file(url, out_path, timeout=300):
    r = requests.get(url, stream=True, timeout=timeout)
    r.raise_for_status()
    content = r.content

    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            tif_names = [n for n in z.namelist() if n.lower().endswith(".tif")]
            if len(tif_names) == 0:
                raise RuntimeError("Zip contains no tif file.")
            with z.open(tif_names[0]) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
    else:
        with open(out_path, "wb") as f:
            f.write(content)


def load_input_table():
    for path in INPUT_CANDIDATES:
        if not path.exists():
            continue

        df = pd.read_csv(path)
        print("\nTrying:", path)
        print("Columns:", list(df.columns))

        lat_col = pick_col(df, ["lat", "latitude", "source_lat", "release_lat", "site_lat"])
        lon_col = pick_col(df, ["lon", "longitude", "source_lon", "release_lon", "site_lon"])
        time_col = pick_col(df, [
            "datetime_utc", "date_time_utc", "event_time_utc",
            "time_utc", "datetime", "date_time", "event_time", "time"
        ])

        if lat_col and lon_col and time_col:
            print("Selected input:", path)
            print("lat:", lat_col)
            print("lon:", lon_col)
            print("time:", time_col)
            return path, df, lat_col, lon_col, time_col

    raise FileNotFoundError("No usable controlled-release table found.")


def add_label(df):
    label_col = pick_col(df, ["label", "classification_label", "true_label", "release_label"])
    release_col = pick_col(df, ["true_release", "release", "is_release", "released"])
    emission_col = pick_col(df, [
        "emission_tph", "true_emission_tph",
        "release_rate_tph", "metered_emission_tph"
    ])

    labels = []

    for _, row in df.iterrows():
        label = None

        if label_col:
            label = parse_label(row[label_col])

        if label is None and release_col:
            label = parse_label(row[release_col])

        if label is None and emission_col:
            label = parse_label(row[emission_col])

        labels.append(label)

    df = df.copy()
    df["label"] = labels
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)
    return df


def filter_valid_rows(df, lat_col, lon_col, time_col):
    df = df.copy()

    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df["_datetime_parsed"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")

    before = len(df)

    df = df.dropna(subset=[lat_col, lon_col, "_datetime_parsed"])
    df = df[df[lat_col].between(-90, 90)]
    df = df[df[lon_col].between(-180, 180)]

    after = len(df)
    print(f"Valid lat/lon/time rows: {after}/{before}")

    return df


def filter_s2_available(df):
    s2_col = pick_col(df, [
        "s2_count", "s2_count_pm1day",
        "sentinel2_count", "sentinel_2_count",
        "s2_available"
    ])

    if s2_col is None:
        print("No S2 availability column found. Will query all valid events.")
        return df

    df = df.copy()
    df[s2_col] = pd.to_numeric(df[s2_col], errors="coerce").fillna(0)

    before = len(df)
    df = df[df[s2_col] > 0].copy()
    after = len(df)

    print(f"S2 available rows using {s2_col}: {after}/{before}")
    return df


def save_index(records):
    out_df = pd.DataFrame(records)
    out_df = out_df.sort_values("export_index")
    out_df.to_csv(OUT_INDEX, index=False)


# =========================
# Main
# =========================
def main():
    initialize_earth_engine()

    input_path, df, lat_col, lon_col, time_col = load_input_table()

    df = add_label(df)
    print("\nAfter label parsing:")
    print(df["label"].value_counts(dropna=False))

    df = filter_valid_rows(df, lat_col, lon_col, time_col)
    df = filter_s2_available(df)
    df = df.reset_index(drop=True)

    print("\nFinal controlled-release S2 candidates:", len(df))
    print("Label counts:")
    print(df["label"].value_counts(dropna=False))

    if len(df) == 0:
        print("No candidates to download.")
        return

    if N_DOWNLOAD is None:
        end_index = len(df)
    else:
        end_index = min(START_INDEX + N_DOWNLOAD, len(df))

    df_batch = df.iloc[START_INDEX:end_index].copy()

    print(f"\nDownloading candidate rows {START_INDEX} to {end_index - 1}")
    print("Batch size:", len(df_batch))

    event_id_col = pick_col(df, ["event_id", "id", "event_name"])
    paper_col = pick_col(df, ["paper", "source_paper", "source_dataset"])
    site_col = pick_col(df, ["site", "site_name", "location_name"])
    emission_col = pick_col(df, [
        "emission_tph", "true_emission_tph",
        "release_rate_tph", "metered_emission_tph"
    ])

    records = []

    for i, row in df_batch.iterrows():
        export_index = START_INDEX + len(records)

        label = int(row["label"])
        lat = float(row[lat_col])
        lon = float(row[lon_col])
        dt = row["_datetime_parsed"]
        dt_str = dt.strftime("%Y-%m-%dT%H:%M:%S")

        if event_id_col:
            event_id = clean_text(row[event_id_col])
        else:
            event_id = f"CR_{export_index:04d}"

        filename = f"CR_S2_patch_{export_index:04d}_label_{label}.tif"
        out_path = OUT_DIR / filename

        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(PATCH_RADIUS).bounds()

        t = ee.Date(dt_str)
        start = t.advance(-1, "day")
        end = t.advance(1, "day")

        s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(point)
            .filterDate(start, end)
            .sort("CLOUDY_PIXEL_PERCENTAGE")
        )

        metadata = {
            "export_index": export_index,
            "filename": filename,
            "relative_path": str(out_path),
            "event_id": event_id,
            "source_dataset": "ControlledRelease",
            "ground_truth_type": "controlled_release",
            "label": label,
            "label_type": "release" if label == 1 else "no_release",
            "sensor": "Sentinel-2",
            "datetime_utc": dt_str,
            "lat": lat,
            "lon": lon,
            "bands": ",".join(BANDS),
            "patch_radius_m": PATCH_RADIUS,
            "scale_m": SCALE,
            "input_csv": str(input_path),
        }

        if paper_col:
            metadata["paper"] = row[paper_col]
        if site_col:
            metadata["site"] = row[site_col]
        if emission_col:
            metadata["emission_tph"] = row[emission_col]

        try:
            s2_size = s2.size().getInfo()
            metadata["s2_count_pm1day"] = s2_size
        except Exception as e:
            print(f"[ERROR] cannot check S2 size for {filename}: {e}")
            metadata["download_status"] = "error_s2_size"
            metadata["error"] = str(e)
            records.append(metadata)
            save_index(records)
            continue

        if s2_size == 0:
            print(f"[NO S2] {filename}")
            metadata["download_status"] = "no_s2_image"
            records.append(metadata)
            save_index(records)
            continue

        img = ee.Image(s2.first()).select(BANDS).clip(region)

        try:
            metadata["s2_image_time"] = ee.Date(img.get("system:time_start")).format(
                "YYYY-MM-dd HH:mm:ss"
            ).getInfo()
        except Exception:
            metadata["s2_image_time"] = ""

        try:
            metadata["s2_cloud_percentage"] = img.get("CLOUDY_PIXEL_PERCENTAGE").getInfo()
        except Exception:
            metadata["s2_cloud_percentage"] = ""

        if out_path.exists():
            print(f"[SKIP] {filename}")
            metadata["download_status"] = "success_existing"
            records.append(metadata)
            save_index(records)
            continue

        print(f"[DOWNLOAD] {filename}")

        try:
            url = img.getDownloadURL({
                "name": f"CR_S2_patch_{export_index:04d}_label_{label}",
                "scale": SCALE,
                "region": region,
                "format": "GEO_TIFF",
                "filePerBand": False,
            })

            download_file(url, out_path)
            metadata["download_status"] = "success"
            print(f"[DONE] {out_path}")

        except Exception as e:
            metadata["download_status"] = "error_download"
            metadata["error"] = str(e)
            print(f"[ERROR] download failed for {filename}: {e}")

        records.append(metadata)
        save_index(records)

        time.sleep(1)

    save_index(records)

    print("\nSaved index:", OUT_INDEX)
    print("Total rows in index:", len(records))


if __name__ == "__main__":
    main()