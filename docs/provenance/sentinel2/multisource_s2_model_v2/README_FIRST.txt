MULTISOURCE SENTINEL-2 MODEL V2
===============================

This package replaces the previous missing-file workflow.

FILES
-----
run_multisource_s2_model_v2.py
install_and_run_multisource_model_v2.sh

WHAT IT DOES
------------
It answers:

Can Sentinel-2 data from different controlled-release sources/sites be fitted
in one model, and can the model predict an unseen site or unseen source?

It automatically:
1. Finds a suitable CSV in your project outputs folder.
2. Detects label/site/source/scene/path columns.
3. Uses existing B11/B12/SWIR features if present.
4. Otherwise reads the GeoTIFF patches and extracts features.
5. Runs leave-one-site-out, leave-one-source-out, and grouped scene splits.
6. Creates outputs/500-506 and figures/507.

EXACT INSTALLATION COMMANDS
---------------------------

After downloading multisource_s2_model_v2.zip, run:

cd ~/Downloads

ZIP_FILE="$(find "$HOME/Downloads" -maxdepth 1 -type f -iname 'multisource_s2_model_v2*.zip' -print -quit)"

echo "$ZIP_FILE"

test -n "$ZIP_FILE" || { echo "ZIP not found in Downloads"; exit 1; }

rm -rf "$HOME/Downloads/multisource_s2_model_v2"

unzip -o "$ZIP_FILE" -d "$HOME/Downloads/multisource_s2_model_v2"

cd "$HOME/Downloads/multisource_s2_model_v2"

bash install_and_run_multisource_model_v2.sh

The installer copies the Python file into:

/Users/happydoraaa/methane_release_project/

Then it activates the existing .venv, runs a self-test, and runs the real model.

MANUAL RUN
----------

cd /Users/happydoraaa/methane_release_project
source .venv/bin/activate

python run_multisource_s2_model_v2.py --self-test

python run_multisource_s2_model_v2.py \
  --project-root /Users/happydoraaa/methane_release_project

USE A SPECIFIC INPUT CSV
------------------------

python run_multisource_s2_model_v2.py \
  --project-root /Users/happydoraaa/methane_release_project \
  --input outputs/390_multisensor_master_manifest_v1.csv

OUTPUTS
-------

outputs/500_multisource_canonical_table_v2.csv
outputs/501_multisource_features_v2.csv
outputs/502_multisource_site_source_summary_v2.csv
outputs/503_multisource_fold_metrics_v2.csv
outputs/504_multisource_predictions_v2.csv
outputs/505_multisource_model_summary_v2.csv
outputs/506_multisource_model_report_v2.txt
figures/507_multisource_loso_balanced_accuracy_v2.png

FIRST RESULT TO COPY BACK
-------------------------

cat /Users/happydoraaa/methane_release_project/outputs/506_multisource_model_report_v2.txt

IF IT SAYS NO RASTER COULD BE READ
----------------------------------

Run:

python - <<'PY'
import pandas as pd
p = "/Users/happydoraaa/methane_release_project/outputs/500_multisource_canonical_table_v2.csv"
x = pd.read_csv(p)
print(x[["sample_id","site_id","source_origin","patch_path_raw","resolved_patch_path"]].head(30).to_string(index=False))
PY

Then copy that table back.
