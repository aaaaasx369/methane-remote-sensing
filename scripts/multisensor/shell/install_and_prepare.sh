#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${1:-/Users/happydoraaa/methane_release_project}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Package: $SCRIPT_DIR"
echo "Project: $PROJECT_ROOT"

test -d "$PROJECT_ROOT" || { echo "Project folder not found"; exit 1; }
test -f "$PROJECT_ROOT/.venv/bin/activate" || { echo ".venv not found"; exit 1; }

cp -f "$SCRIPT_DIR/prepare_and_finalize_five_sites.py" "$PROJECT_ROOT/"
cp -f "$SCRIPT_DIR/download_methaneair_reference_negatives.py" "$PROJECT_ROOT/"
cp -f "$SCRIPT_DIR/run_multisource_s2_model_v2.py" "$PROJECT_ROOT/"

cd "$PROJECT_ROOT"
source .venv/bin/activate
python -m pip install -q pandas numpy scikit-learn matplotlib rasterio earthengine-api requests

python run_multisource_s2_model_v2.py --self-test
python prepare_and_finalize_five_sites.py --project-root "$PROJECT_ROOT"

echo
echo "Preparation complete. Review:"
echo "$PROJECT_ROOT/outputs/543_five_site_prepare_report_v1.txt"
echo
echo "Then run the Earth Engine download step from README_FIRST.md."
