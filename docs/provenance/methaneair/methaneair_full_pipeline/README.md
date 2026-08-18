# MethaneAIR ground truth + images + Sentinel-2 pipeline

## What this package does

1. Downloads the official MethaneAIR L4 point-source inventory that your
   Earth Engine account is allowed to access.
2. Downloads the official MethaneAIR L3 image inventory.
3. Optionally downloads an L3 XCH4 patch around every L4 plume.
4. Adds existing controlled/physical-release positives and negatives from the
   current unified master.
5. Creates MethaneAIR same-site temporal negative candidates at +1, +3, +7
   and +14 days.
6. Searches public Sentinel-2 L2A imagery for every usable record.
7. Downloads t0, t90 and t360 image patches with:
   B02, B03, B04, B08, B11, B12 and a separate SCL file.
8. Calculates SCL clear fraction and QA pass.

## Important label rule

- MethaneAIR L4 detections are positive observational labels.
- Existing physical/controlled-release label 0 records are confirmed negatives.
- A date with no MethaneAIR plume in the public L4 table is NOT automatically
  negative.
- Generated +1/+3/+7/+14 day negatives stay `candidate_unconfirmed` until:
  - known release exclusion is complete;
  - known plume exclusion is complete;
  - nearby plume exclusion is complete;
  - SCL cloud/snow QA passes.

## Server installation

From your Mac:

```bash
scp methaneair_full_pipeline.zip \
yunjung1@fir.alliancecan.ca:/project/6002520/yunjung1/MethaneFuse/
```

On Fir:

```bash
cd /project/6002520/yunjung1/MethaneFuse
unzip -o methaneair_full_pipeline.zip
source /project/6002520/yunjung1/venvs/carbonmapper311/bin/activate
```

Check required packages:

```bash
python - <<'PY'
import pandas, numpy, requests, rasterio
print("Core packages OK")
try:
    import ee
    print("Earth Engine package OK")
except Exception as exc:
    print("Earth Engine package missing:", exc)
PY
```

## Step 1 — verify MethaneAIR access and export ground truth

Set the Earth Engine project:

```bash
export EE_PROJECT="methane-release-gee"
```

Run inventory only first:

```bash
python methaneair_full_pipeline/export_methaneair_gee.py \
  --project-root /project/6002520/yunjung1/MethaneFuse \
  --ee-project "$EE_PROJECT"
```

Expected outputs:

```text
data/methaneair_full/methaneair_l4_points.csv
data/methaneair_full/methaneair_l4_points.geojson
data/methaneair_full/methaneair_l3_inventory.csv
```

If access is denied, the script writes:

```text
data/methaneair_full/ACCESS_REQUIRED.txt
```

The official publisher dataset requires approved access. Complete the
MethaneSAT request form, then rerun.

## Step 2 — download MethaneAIR L3 image patches

Start with five records as a test:

```bash
python methaneair_full_pipeline/export_methaneair_gee.py \
  --project-root /project/6002520/yunjung1/MethaneFuse \
  --ee-project "$EE_PROJECT" \
  --download-l3-patches \
  --max-points 5 \
  --resume
```

After the five-record test succeeds, download all permitted patches:

```bash
python methaneair_full_pipeline/export_methaneair_gee.py \
  --project-root /project/6002520/yunjung1/MethaneFuse \
  --ee-project "$EE_PROJECT" \
  --download-l3-patches \
  --resume
```

## Step 3 — build positives, confirmed negatives and negative candidates

```bash
python methaneair_full_pipeline/build_methaneair_ground_truth.py \
  --project-root /project/6002520/yunjung1/MethaneFuse
```

Outputs:

```text
data/methaneair_full/ground_truth_confirmed.csv
data/methaneair_full/ground_truth_negative_candidates.csv
data/methaneair_full/ground_truth_all.csv
data/methaneair_full/ground_truth_summary.csv
```

## Step 4 — test Sentinel-2 matching on five records

This Sentinel-2 route does not require Earth Engine authentication.

```bash
python methaneair_full_pipeline/download_sentinel2_matches.py \
  --project-root /project/6002520/yunjung1/MethaneFuse \
  --include-candidate-negatives \
  --max-records 5 \
  --resume
```

## Step 5 — download all corresponding Sentinel-2 images

Use Slurm for the full run:

```bash
cd /project/6002520/yunjung1/MethaneFuse
mkdir -p logs
sbatch methaneair_full_pipeline/run_sentinel2_download.sbatch
```

Monitor:

```bash
squeue -u "$USER"
tail -f logs/mair_s2_dl_<JOBID>.out
```

Outputs:

```text
data/methaneair_full/sentinel2/*.tif
data/methaneair_full/sentinel2_match_long.csv
data/methaneair_full/sentinel2_temporal_manifest.csv
data/methaneair_full/sentinel2_download_summary.csv
```

## Check final counts

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path

root = Path(
    "/project/6002520/yunjung1/MethaneFuse/data/methaneair_full"
)

for name in [
    "ground_truth_summary.csv",
    "sentinel2_download_summary.csv",
]:
    path = root / name
    print("\n", path)
    if path.exists():
        print(pd.read_csv(path).to_string(index=False))
    else:
        print("NOT CREATED")
PY
```
