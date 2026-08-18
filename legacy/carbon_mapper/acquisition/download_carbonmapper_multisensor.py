import os
import re
import time
import requests
import pandas as pd
from pathlib import Path
from urllib.parse import urlparse

# ============================================================
# SETTINGS
# ============================================================

INPUT_CSV = "carbon_mapper_all_CH4_plumes.csv"

# FIRST TEST:
# download 100 cases from EACH sensor
MAX_PER_SENSOR = 100

# Download both products when available
DOWNLOAD_PLUME_TIF = True
DOWNLOAD_CON_TIF = True

# Sensors we want
SENSORS = [
    "tan",
    "GAO",
    "ang",
    "emi",
    "av3",
]

SENSOR_NAMES = {
    "tan": "Tanager",
    "GAO": "GAO",
    "ang": "AVIRIS-NG",
    "emi": "EMIT",
    "av3": "AVIRIS-3",
}

OUT_ROOT = Path("carbonmapper_multisensor_download")

TOKEN = os.environ.get("CARBON_MAPPER_TOKEN")

if not TOKEN:
    raise RuntimeError(
        "CARBON_MAPPER_TOKEN is not set.\n"
        "Run:\n"
        "export CARBON_MAPPER_TOKEN='YOUR_TOKEN'"
    )

OUT_ROOT.mkdir(exist_ok=True)

# ============================================================
# HELPERS
# ============================================================

def valid_url(x):
    if pd.isna(x):
        return False

    x = str(x).strip()

    return x.startswith("http://") or x.startswith("https://")


def safe_name(x):
    x = str(x)

    return re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        x
    )[:150]


def download_file(url, output_path):

    if output_path.exists():
        print("  EXISTS:", output_path.name)
        return True

    headers = {
        "User-Agent": "UAlberta-methane-research/1.0"
    }

    # Direct Carbon Mapper API URLs can use token.
    # Storage/signed URLs are first tried without Authorization.
    host = urlparse(url).netloc.lower()

    if "api.carbonmapper.org" in host:
        headers["Authorization"] = f"Bearer {TOKEN}"

    try:

        with requests.get(
            url,
            headers=headers,
            stream=True,
            timeout=180,
            allow_redirects=True
        ) as r:

            # Retry with token if needed
            if r.status_code in (401, 403):

                headers["Authorization"] = f"Bearer {TOKEN}"

                r = requests.get(
                    url,
                    headers=headers,
                    stream=True,
                    timeout=180,
                    allow_redirects=True
                )

            r.raise_for_status()

            total = int(
                r.headers.get(
                    "content-length",
                    0
                )
            )

            if total > 0:
                print(
                    f"  downloading {output_path.name} "
                    f"({total / 1024 / 1024:.1f} MB)"
                )
            else:
                print(
                    f"  downloading {output_path.name}"
                )

            tmp_path = output_path.with_suffix(
                output_path.suffix + ".part"
            )

            with open(tmp_path, "wb") as f:

                for chunk in r.iter_content(
                    chunk_size=1024 * 1024
                ):

                    if chunk:
                        f.write(chunk)

            tmp_path.rename(output_path)

        return True

    except Exception as e:

        print(
            "  DOWNLOAD FAILED:",
            str(e)
        )

        return False


# ============================================================
# LOAD MASTER INVENTORY
# ============================================================

df = pd.read_csv(INPUT_CSV)

print("=" * 70)
print("CARBON MAPPER MULTI-SENSOR DOWNLOADER")
print("=" * 70)

print("Input rows:", len(df))
print("Columns:", list(df.columns))


# ============================================================
# KEEP CH4 + REQUIRED SENSOR
# ============================================================

df = df[
    df["instrument"].isin(SENSORS)
].copy()


# ============================================================
# REQUIRE PLUME TIFF
# ============================================================

df["has_plume_tif"] = df["plume_tif"].apply(
    valid_url
)

df["has_con_tif"] = df["con_tif"].apply(
    valid_url
)

usable = df[
    df["has_plume_tif"]
].copy()


print("")
print("Usable plume TIFF rows:", len(usable))


# ============================================================
# SELECT REPRESENTATIVE CASES
#
# Prefer records with:
#  - plume TIFF
#  - concentration TIFF
#  - emission rate
#
# Sort by emission strength and take evenly spaced samples.
# This avoids picking only strongest/weakest plumes.
# ============================================================

selected_groups = []

for sensor in SENSORS:

    g = usable[
        usable["instrument"] == sensor
    ].copy()

    # Prefer complete cases
    complete = g[
        g["has_con_tif"]
    ].copy()

    # Prefer cases with emission rates
    with_flux = complete[
        complete["emission_auto_kg_hr"].notna()
    ].copy()

    if len(with_flux) >= MAX_PER_SENSOR:
        pool = with_flux

    elif len(complete) >= MAX_PER_SENSOR:
        pool = complete

    else:
        pool = g

    # Sort by flux where possible
    pool = pool.sort_values(
        "emission_auto_kg_hr",
        na_position="last"
    ).reset_index(drop=True)

    n = min(
        MAX_PER_SENSOR,
        len(pool)
    )

    if n == 0:
        continue

    # evenly spaced across full pool
    if n == 1:
        indices = [0]
    else:
        indices = [
            round(
                i * (len(pool) - 1) / (n - 1)
            )
            for i in range(n)
        ]

    picked = pool.iloc[
        sorted(set(indices))
    ].copy()

    picked["download_sensor_name"] = SENSOR_NAMES[sensor]

    selected_groups.append(picked)

    print(
        f"{sensor:4s} / "
        f"{SENSOR_NAMES[sensor]:10s}: "
        f"available={len(g):5d}, "
        f"complete={len(complete):5d}, "
        f"selected={len(picked):3d}"
    )


selected = pd.concat(
    selected_groups,
    ignore_index=True
)


# ============================================================
# SAVE SELECTION MANIFEST BEFORE DOWNLOAD
# ============================================================

manifest_path = OUT_ROOT / "selected_cases_manifest.csv"

selected.to_csv(
    manifest_path,
    index=False
)

print("")
print("Selected total:", len(selected))
print("Manifest:", manifest_path)


# ============================================================
# DOWNLOAD
# ============================================================

results = []

for i, row in selected.iterrows():

    sensor = str(
        row["instrument"]
    )

    sensor_name = SENSOR_NAMES[
        sensor
    ]

    sensor_dir = OUT_ROOT / sensor_name

    sensor_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    plume_id = safe_name(
        row.get(
            "plume_id",
            f"row_{i}"
        )
    )

    timestamp = safe_name(
        row.get(
            "scene_timestamp",
            "unknown_time"
        )
    )

    base = (
        f"{sensor}_{plume_id}_{timestamp}"
    )

    print("")
    print(
        f"[{i + 1}/{len(selected)}] "
        f"{sensor_name} | "
        f"{plume_id}"
    )

    plume_ok = None
    con_ok = None

    # --------------------------------------------------------
    # PLUME TIFF
    # --------------------------------------------------------

    if DOWNLOAD_PLUME_TIF and valid_url(
        row["plume_tif"]
    ):

        plume_path = sensor_dir / (
            base + "__plume.tif"
        )

        plume_ok = download_file(
            str(row["plume_tif"]),
            plume_path
        )

    # --------------------------------------------------------
    # CONCENTRATION TIFF
    # --------------------------------------------------------

    if DOWNLOAD_CON_TIF and valid_url(
        row["con_tif"]
    ):

        con_path = sensor_dir / (
            base + "__con.tif"
        )

        con_ok = download_file(
            str(row["con_tif"]),
            con_path
        )

    results.append({
        "instrument": sensor,
        "sensor_name": sensor_name,
        "plume_id": row.get("plume_id"),
        "scene_timestamp": row.get("scene_timestamp"),
        "emission_auto_kg_hr": row.get("emission_auto_kg_hr"),
        "plume_download_ok": plume_ok,
        "con_download_ok": con_ok,
    })

    # be polite to server
    time.sleep(0.1)


# ============================================================
# SAVE DOWNLOAD REPORT
# ============================================================

report = pd.DataFrame(results)

report_path = OUT_ROOT / "download_report.csv"

report.to_csv(
    report_path,
    index=False
)


print("")
print("=" * 70)
print("DOWNLOAD COMPLETE")
print("=" * 70)

print(
    report.groupby(
        "sensor_name"
    )[
        [
            "plume_download_ok",
            "con_download_ok"
        ]
    ].sum()
)

print("")
print(
    "Successful plume TIFFs:",
    int(
        report["plume_download_ok"]
        .fillna(False)
        .sum()
    )
)

print(
    "Successful concentration TIFFs:",
    int(
        report["con_download_ok"]
        .fillna(False)
        .sum()
    )
)

print("")
print("Output folder:")
print(OUT_ROOT.resolve())

print("")
print("Report:")
print(report_path.resolve())

