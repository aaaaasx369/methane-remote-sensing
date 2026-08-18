from pathlib import Path
import io
import re
import time
import zipfile
import requests
import pandas as pd
import ee


# =========================
# Settings
# =========================
PROJECT = "methane-release-gee"

PATCH_RADIUS = 1000  # meters
SCALE = 30           # Landsat resolution

# Landsat Collection 2 Level 2 surface reflectance bands
LANDSAT_BANDS = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]

# Rename to common names, so it matches S2-style logic:
# blue, green, red, NIR, SWIR1, SWIR2
COMMON_BANDS = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]

START_INDEX = 0

# 先測試 20 張；成功後再改成 None 下載全部
N_DOWNLOAD = None

INPUT_PATH = Path("outputs/12_dataset_candidate_events_with_latlon.csv")

OUT_DIR = Path("sample_patches/controlled_release_landsat")
OUT_INDEX = Path("outputs/30_controlled_release_landsat_patch_index.csv")

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_INDEX.parent.mkdir(parents=True, exist_ok=True)


# =========================
# Helper functions
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


def clean_text(x):
    x = str(x)
    x = re.sub(r"[^A-Za-z0-9_\-]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_")


def download_file(url, out_path, timeout=300):
    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()
    content = response.content

    # Earth Engine sometimes returns zipped GeoTIFF
    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            tif_names = [
                name for name in z.namelist()
                if name.lower().endswith(".tif")
            ]

            if len(tif_names) == 0:
                raise RuntimeError("Downloaded zip file contains no .tif file.")

            with z.open(tif_names[0]) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
    else:
        with open(out_path, "wb") as f:
            f.write(content)


def prepare_landsat_image_collection(point, start, end):
    """
    Merge Landsat 8 and Landsat 9 Level-2 SR collections.
    Select comparable optical + SWIR bands and rename them.
    """
    l8 = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(point)
        .filterDate(start, end)
        .select(LANDSAT_BANDS, COMMON_BANDS)
        .map(lambda img: img.set("landsat_sensor", "Landsat-8"))
    )

    l9 = (
        ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        .filterBounds(point)
        .filterDate(start, end)
        .select(LANDSAT_BANDS, COMMON_BANDS)
        .map(lambda img: img.set("landsat_sensor", "Landsat-9"))
    )

    merged = l8.merge(l9).sort("CLOUD_COVER")

    return merged


def main():
    initialize_earth_engine()

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_PATH}. "
            "Run fix_controlled_release_candidate_latlon.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    print("Input:", INPUT_PATH)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    # Basic cleaning
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df["_datetime_parsed"] = pd.to_datetime(
        df["datetime_utc"],
        utc=True,
        errors="coerce"
    )

    before = len(df)

    df = df.dropna(subset=["lat", "lon", "label", "_datetime_parsed"])
    df = df[df["lat"].between(-90, 90)]
    df = df[df["lon"].between(-180, 180)]
    df = df[df["label"].isin([0, 1])]
    df["label"] = df["label"].astype(int)

    after = len(df)

    print(f"Valid lat/lon/time/label rows: {after}/{before}")

    if len(df) == 0:
        print("No valid events.")
        return

    print("\nLabel counts:")
    print(df["label"].value_counts())

    if N_DOWNLOAD is None:
        end_index = len(df)
    else:
        end_index = min(START_INDEX + N_DOWNLOAD, len(df))

    df_batch = df.iloc[START_INDEX:end_index].copy()

    print(f"\nDownloading candidate rows {START_INDEX} to {end_index - 1}")
    print("Batch size:", len(df_batch))

    records = []

    for local_i, (_, row) in enumerate(df_batch.iterrows()):
        export_index = START_INDEX + local_i

        label = int(row["label"])
        lat = float(row["lat"])
        lon = float(row["lon"])
        dt = row["_datetime_parsed"]
        dt_str = dt.strftime("%Y-%m-%dT%H:%M:%S")

        event_id = clean_text(row.get("event_id", f"CR_{export_index:04d}"))

        filename = f"CR_Landsat_patch_{export_index:04d}_label_{label}.tif"
        out_path = OUT_DIR / filename

        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(PATCH_RADIUS).bounds()

        t = ee.Date(dt_str)
        start = t.advance(-1, "day")
        end = t.advance(1, "day")

        landsat = prepare_landsat_image_collection(point, start, end)

        metadata = {
            "export_index": export_index,
            "filename": filename,
            "relative_path": str(out_path),
            "event_id": event_id,
            "source_dataset": "ControlledRelease",
            "ground_truth_type": "controlled_release",
            "label": label,
            "label_type": "release" if label == 1 else "no_release",
            "sensor": "Landsat-8/9",
            "datetime_utc": dt_str,
            "lat": lat,
            "lon": lon,
            "bands": ",".join(COMMON_BANDS),
            "original_landsat_bands": ",".join(LANDSAT_BANDS),
            "patch_radius_m": PATCH_RADIUS,
            "scale_m": SCALE,
            "input_csv": str(INPUT_PATH),
        }

        for optional_col in [
            "paper",
            "site_name",
            "satellite_from_paper",
            "emission_tph_mean",
            "emission_tph_median",
            "emission_tph_max",
            "l8_count",
            "l9_count",
            "usable_landsat",
            "usable_landsat8",
            "usable_landsat9",
        ]:
            if optional_col in row.index:
                metadata[optional_col] = row[optional_col]

        try:
            landsat_size = landsat.size().getInfo()
            metadata["landsat_count_pm1day"] = landsat_size
        except Exception as e:
            print(f"[ERROR] cannot check Landsat size for {filename}: {e}")
            metadata["download_status"] = "error_landsat_size"
            metadata["error"] = str(e)
            records.append(metadata)
            pd.DataFrame(records).to_csv(OUT_INDEX, index=False)
            continue

        if landsat_size == 0:
            print(f"[NO LANDSAT] {filename}")
            metadata["download_status"] = "no_landsat_image"
            records.append(metadata)
            pd.DataFrame(records).to_csv(OUT_INDEX, index=False)
            continue

        img = ee.Image(landsat.first()).clip(region)

        try:
            metadata["landsat_image_time"] = ee.Date(
                img.get("system:time_start")
            ).format("YYYY-MM-dd HH:mm:ss").getInfo()
        except Exception:
            metadata["landsat_image_time"] = ""

        try:
            metadata["landsat_cloud_cover"] = img.get("CLOUD_COVER").getInfo()
        except Exception:
            metadata["landsat_cloud_cover"] = ""

        try:
            metadata["landsat_sensor"] = img.get("landsat_sensor").getInfo()
        except Exception:
            metadata["landsat_sensor"] = ""

        if out_path.exists():
            print(f"[SKIP] {filename}")
            metadata["download_status"] = "success_existing"
            records.append(metadata)
            pd.DataFrame(records).to_csv(OUT_INDEX, index=False)
            continue

        print(f"[DOWNLOAD] {filename}")

        try:
            url = img.getDownloadURL({
                "name": f"CR_Landsat_patch_{export_index:04d}_label_{label}",
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
        pd.DataFrame(records).to_csv(OUT_INDEX, index=False)

        time.sleep(1)

    pd.DataFrame(records).to_csv(OUT_INDEX, index=False)

    print("\nSaved index:", OUT_INDEX)
    print("Total rows in index:", len(records))

    if len(records) > 0:
        out_df = pd.DataFrame(records)
        print("\nDownload status:")
        print(out_df["download_status"].value_counts(dropna=False))

        if "label" in out_df.columns:
            print("\nLabel counts in index:")
            print(out_df["label"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
