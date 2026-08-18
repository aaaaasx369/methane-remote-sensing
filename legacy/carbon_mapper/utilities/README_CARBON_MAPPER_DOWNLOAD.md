# Carbon Mapper controlled-release download starter

This starter queries Carbon Mapper's public plume endpoint around two controlled-release sites:

- Ehrenberg, Arizona (2021)
- Casa Grande, Arizona (2022)

## 1. Install the only required package

```bash
python -m pip install requests
```

## 2. Run metadata audit first

```bash
python download_carbon_mapper_controlled_release.py \
  --sites carbon_mapper_controlled_release_sites.csv \
  --output data/carbon_mapper_controlled_release
```

Expected outputs:

```text
data/carbon_mapper_controlled_release/
├── carbon_mapper_all_plumes.csv
├── carbon_mapper_download_summary.json
├── Ehrenberg_2021/
│   └── metadata/
└── Casa_Grande_2022/
    └── metadata/
```

First inspect:

```bash
column -s, -t < data/carbon_mapper_controlled_release/carbon_mapper_all_plumes.csv | less -S
```

Or:

```bash
python - <<'PY'
import pandas as pd

p = "data/carbon_mapper_controlled_release/carbon_mapper_all_plumes.csv"
df = pd.read_csv(p)

print("Rows:", len(df))
print("\nSites:")
print(df["site_id"].value_counts(dropna=False))
print("\nInstruments:")
print(df["instrument"].value_counts(dropna=False))
print("\nDates:")
print(df[["site_id", "scene_timestamp", "plume_id", "plume_quality"]].to_string(index=False))
print("\nAvailable files:")
for c in ["plume_tif", "con_tif", "rgb_tif", "plume_png", "rgb_png"]:
    print(c, int(df[c].notna().sum()) if c in df.columns else 0)
PY
```

## 3. Download the useful GeoTIFF products

```bash
python download_carbon_mapper_controlled_release.py \
  --sites carbon_mapper_controlled_release_sites.csv \
  --output data/carbon_mapper_controlled_release \
  --download \
  --assets plume_tif con_tif rgb_tif
```

Safer small test:

```bash
python download_carbon_mapper_controlled_release.py \
  --sites carbon_mapper_controlled_release_sites.csv \
  --output data/carbon_mapper_controlled_release_test \
  --download \
  --assets plume_tif
```

## 4. Verify downloaded TIFF files

```bash
python - <<'PY'
from pathlib import Path
import rasterio

root = Path("data/carbon_mapper_controlled_release")
tifs = sorted(root.rglob("*.tif"))

print("TIFF count:", len(tifs))

for p in tifs:
    try:
        with rasterio.open(p) as src:
            print(
                p,
                "bands=", src.count,
                "shape=", (src.height, src.width),
                "crs=", src.crs,
                "dtype=", src.dtypes,
                "nodata=", src.nodata,
            )
    except Exception as exc:
        print("ERROR", p, exc)
PY
```

Install rasterio only if it is missing:

```bash
python -m pip install rasterio
```

## Interpretation

- `plume_tif`: plume-level georeferenced product; useful as Carbon Mapper reference plume support.
- `con_tif`: methane concentration/retrieval image when exposed by the API.
- `rgb_tif`: simultaneous RGB context when exposed by the API.
- A plume query returns detected plumes, so it is positive-biased.
- A date with no returned plume is not automatically a valid negative. Scene coverage, cloud, and quality need a separate audit.
- Do not treat these products as the original full hyperspectral radiance cube.
- Retain attribution such as `Source: Carbon Mapper`.
