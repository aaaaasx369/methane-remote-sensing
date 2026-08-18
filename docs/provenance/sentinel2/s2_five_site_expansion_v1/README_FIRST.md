# Five-site Sentinel-2 expansion

The three selected MethaneAIR sites have passed the positive audit:

- MethaneAIR_site_073
- MethaneAIR_site_102
- MethaneAIR_site_120

Each now has at least two independent positive Sentinel-2 scenes.

## Scientific structure

Final five sites:

1. Casa Grande — metered controlled release
2. Ehrenberg — metered controlled release
3. MethaneAIR site 073 — observational plume
4. MethaneAIR site 102 — observational plume
5. MethaneAIR site 120 — observational plume

The new label-0 scenes are called **no-known-plume reference negatives**.
They are not confirmed zero-emission measurements.

---

## Installation

Download and unzip `s2_five_site_expansion_v1.zip`.

Then:

```bash
cd ~/Downloads

ZIP_FILE="$(find "$HOME/Downloads" -maxdepth 1 -type f \
  -iname 's2_five_site_expansion_v1*.zip' -print -quit)"

test -n "$ZIP_FILE" || {
  echo "ZIP not found"
  exit 1
}

rm -rf "$HOME/Downloads/s2_five_site_expansion_v1"

unzip -o "$ZIP_FILE" \
  -d "$HOME/Downloads/s2_five_site_expansion_v1"

cp "$HOME/Downloads/s2_five_site_expansion_v1/"*.py \
  /Users/happydoraaa/methane_release_project/
```

Activate the environment:

```bash
cd /Users/happydoraaa/methane_release_project
source .venv/bin/activate

python -m pip install earthengine-api requests rasterio pandas numpy
```

Authenticate Earth Engine when needed:

```bash
earthengine authenticate
export EE_PROJECT="YOUR_GOOGLE_CLOUD_PROJECT_ID"
```

---

## Step 1 — search only

Always inspect candidates before downloading:

```bash
python find_and_download_s2_reference_negatives_v1.py \
  --project-root /Users/happydoraaa/methane_release_project \
  --search-only
```

Outputs:

```text
outputs/515_s2_negative_candidates_v1.csv
outputs/516_s2_negative_selected_v1.csv
outputs/518_s2_negative_download_report_v1.txt
```

Inspect:

```bash
cat outputs/518_s2_negative_download_report_v1.txt
```

A good result should contain at least 8 selected candidates per site.

Check candidate dates:

```bash
python - <<'PY'
import pandas as pd

x = pd.read_csv("outputs/516_s2_negative_selected_v1.csv")
print(
    x[
        [
            "site_id",
            "s2_time_utc",
            "clear_fraction",
            "nearest_known_local_plume_days",
            "seasonal_distance_days",
            "download_priority",
            "selected_for_download",
        ]
    ].to_string(index=False)
)
PY
```

---

## Step 2 — download eight references per site

After candidate review:

```bash
python find_and_download_s2_reference_negatives_v1.py \
  --project-root /Users/happydoraaa/methane_release_project
```

Outputs:

```text
outputs/517_s2_negative_manifest_v1.csv
patches/s2_matched_negatives_v1/
```

Audit downloads:

```bash
cat outputs/518_s2_negative_download_report_v1.txt
```

---

## Step 3 — combine into five sites

```bash
python build_five_site_multisource_manifest_v1.py \
  --project-root /Users/happydoraaa/methane_release_project
```

Outputs:

```text
outputs/519_five_site_multisource_manifest_v1.csv
outputs/520_five_site_multisource_audit_v1.csv
outputs/521_five_site_multisource_report_v1.txt
```

Expected headline:

```text
Unique sites: 5
Unique sources: at least 2
```

Inspect:

```bash
cat outputs/521_five_site_multisource_report_v1.txt
```

---

## Step 4 — retrain

```bash
python run_multisource_s2_model_v2.py \
  --project-root /Users/happydoraaa/methane_release_project \
  --input outputs/519_five_site_multisource_manifest_v1.csv
```

Open:

```bash
cat outputs/506_multisource_model_report_v2.txt
```

Primary result:

```text
leave_one_site_out
```

Secondary source-transfer result:

```text
leave_one_source_out
```

The grouped random result remains an optimistic reference only.
