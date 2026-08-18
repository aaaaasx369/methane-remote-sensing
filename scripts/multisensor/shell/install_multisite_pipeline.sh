#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${1:-/Users/happydoraaa/methane_release_project}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

test -d "$PROJECT_ROOT" || { echo "Project not found: $PROJECT_ROOT"; exit 1; }
test -f "$PROJECT_ROOT/.venv/bin/activate" || { echo ".venv not found"; exit 1; }

cp -f "$SCRIPT_DIR/multisite_pipeline.py" "$PROJECT_ROOT/multisite_pipeline.py"
chmod +x "$PROJECT_ROOT/multisite_pipeline.py"

cd "$PROJECT_ROOT"
source .venv/bin/activate

python -m pip install -q pandas numpy rasterio scikit-learn matplotlib
python -m py_compile multisite_pipeline.py

echo "Installed: $PROJECT_ROOT/multisite_pipeline.py"
echo
echo "Run local steps:"
echo "python multisite_pipeline.py all --project-root $PROJECT_ROOT"
echo
echo "Run full Earth Engine steps:"
echo "export EE_PROJECT='methane-release-gee'"
echo "python multisite_pipeline.py all --project-root $PROJECT_ROOT --use-ee --use-ee-landsat --ee-project \"\$EE_PROJECT\""
