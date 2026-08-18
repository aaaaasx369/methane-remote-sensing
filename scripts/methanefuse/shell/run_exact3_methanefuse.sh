#!/bin/bash
#SBATCH --job-name=mf_exact3
#SBATCH --account=def-juliana2
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/mf_exact3_%j.out
#SBATCH --error=logs/mf_exact3_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

module purge
module load python/3.11
module load scipy-stack/2025a

VENV=/project/def-juliana2/yunjung1/venvs/methanefuse

if [ ! -f "$VENV/bin/activate" ]; then
    echo "ERROR: virtual environment not found: $VENV"
    exit 1
fi

source "$VENV/bin/activate"

echo "============================================================"
echo "MethaneFuse exact-3 controlled-release inference"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
python --version
echo "============================================================"

python - <<'PY'
import torch

print("torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot access the allocated GPU.")
PY

CSV=data/custom/exact3_s2_controlled_release.csv

if [ ! -f "$CSV" ]; then
    echo "ERROR: evaluation CSV not found: $CSV"
    exit 1
fi

python - <<'PY'
from pathlib import Path
import pandas as pd

path = Path("data/custom/exact3_s2_controlled_release.csv")
df = pd.read_csv(path)

print("\nEvaluation manifest:")
print(df[["id", "sample_id", "label", "metered_release_rate_kg_hr"]])

if len(df) != 3:
    raise RuntimeError(f"Expected 3 rows, found {len(df)}")

for column in ["s2_0_path", "s2_90_path", "s2_360_path"]:
    missing = [
        value for value in df[column]
        if not Path(str(value)).exists()
    ]

    if missing:
        raise FileNotFoundError(
            f"{column} contains missing files:\n"
            + "\n".join(missing)
        )

print("\nAll image paths verified.")
PY

CHECKPOINT=$(
    find checkpoints . \
        -type f \
        \( \
            -name "methanefuse_cls_480m.pt" \
            -o -name "stage2_classification_480m.pt" \
            -o -name "*classification*480m*.pt" \
        \) \
        -print 2>/dev/null |
    head -1
)

WEIGHTS=$(
    find weights . \
        -type f \
        -name "panopticon_vitb14_teacher.pth" \
        -print 2>/dev/null |
    head -1
)

if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: 480 m classification checkpoint not found."
    echo "Available .pt files:"
    find . -type f -name "*.pt" -print 2>/dev/null | head -50
    exit 1
fi

if [ -z "$WEIGHTS" ]; then
    echo "ERROR: panopticon_vitb14_teacher.pth not found."
    echo "Available .pth files:"
    find . -type f -name "*.pth" -print 2>/dev/null | head -50
    exit 1
fi

echo
echo "Checkpoint: $CHECKPOINT"
echo "Backbone weights: $WEIGHTS"
echo

nvidia-smi

python -m py_compile scripts/eval/evaluate_classification.py

echo
echo "Starting inference..."
echo

python examples/inference_480m.py \
    --eval_csv "$CSV" \
    --checkpoint "$CHECKPOINT" \
    --weights "$WEIGHTS" \
    --stage b \
    --batch_size 1 \
    --num_workers 0 \
    --device cuda \
    --row_fusion_mode max \
    --output_json results/custom/exact3_s2_controlled_release.json \
    --no_normalize_columns

echo
echo "Inference completed successfully."
echo "Result:"
cat results/custom/exact3_s2_controlled_release.json
