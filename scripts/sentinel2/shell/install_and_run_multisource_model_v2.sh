#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/Users/happydoraaa/methane_release_project"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE_SCRIPT="$SCRIPT_DIR/run_multisource_s2_model_v2.py"
TARGET_SCRIPT="$PROJECT_ROOT/run_multisource_s2_model_v2.py"

echo "Package folder: $SCRIPT_DIR"
echo "Project folder: $PROJECT_ROOT"

if [ ! -d "$PROJECT_ROOT" ]; then
  echo "ERROR: project folder does not exist:"
  echo "$PROJECT_ROOT"
  exit 1
fi

if [ ! -f "$SOURCE_SCRIPT" ]; then
  echo "ERROR: main Python file is missing:"
  echo "$SOURCE_SCRIPT"
  exit 1
fi

cp -f "$SOURCE_SCRIPT" "$TARGET_SCRIPT"
chmod +x "$TARGET_SCRIPT"

cd "$PROJECT_ROOT"

if [ ! -f ".venv/bin/activate" ]; then
  echo "ERROR: .venv was not found in $PROJECT_ROOT"
  exit 1
fi

source .venv/bin/activate

echo
echo "Checking Python packages..."
python - <<'PY'
missing = []
for name in ["pandas", "numpy", "sklearn", "matplotlib", "rasterio"]:
    try:
        __import__(name)
    except Exception:
        missing.append(name)
if missing:
    print("MISSING:", " ".join(missing))
    raise SystemExit(2)
print("All required packages are available.")
PY

echo
echo "Running built-in self-test..."
python "$TARGET_SCRIPT" --self-test

echo
echo "Running the real project analysis..."
python "$TARGET_SCRIPT" --project-root "$PROJECT_ROOT"

echo
echo "DONE"
echo "Open this report:"
echo "$PROJECT_ROOT/outputs/506_multisource_model_report_v2.txt"
