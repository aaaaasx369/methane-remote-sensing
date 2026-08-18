#!/bin/bash
#SBATCH --job-name=mf-eval-f1b
#SBATCH --account=def-juliana2
#SBATCH --output=logs/mf_eval_f1b_%j.out
#SBATCH --error=logs/mf_eval_f1b_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:h100:1

set -euo pipefail

REPO_ROOT="/project/6002520/yunjung1/MethaneFuse"
VENV_PATH="/project/6002520/yunjung1/venvs/methanefuse"

TEST_CSV="$REPO_ROOT/data/custom/two_negative_group_cv/fold_1_baseline/test.csv"

CHECKPOINT="$REPO_ROOT/checkpoints/two_negative_group_cv/fold_1_baseline/train__val__head/ckpt_best_test.pth"

WEIGHTS="$REPO_ROOT/weights/panopticon_vitb14_teacher.pth"

OUTPUT_JSON="$REPO_ROOT/results/two_negative_group_cv/fold_1_baseline_heldout.json"

cd "$REPO_ROOT"

mkdir -p \
  logs \
  results/two_negative_group_cv

source "$VENV_PATH/bin/activate"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

for file in \
  "$TEST_CSV" \
  "$CHECKPOINT" \
  "$WEIGHTS"
do
  if [[ ! -f "$file" ]]; then
    echo "Missing required file: $file" >&2
    exit 2
  fi
done

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable.")

print("GPU:", torch.cuda.get_device_name(0))
PY

python scripts/eval/evaluate_classification.py \
  --eval_csv "$TEST_CSV" \
  --checkpoint "$CHECKPOINT" \
  --weights "$WEIGHTS" \
  --stage b \
  --lora_rank 4 \
  --lora_alpha 8 \
  --batch_size 4 \
  --num_workers 4 \
  --device cuda \
  --row_fusion_mode max \
  --output_json "$OUTPUT_JSON"

echo "Evaluation completed."
echo "Result: $OUTPUT_JSON"
