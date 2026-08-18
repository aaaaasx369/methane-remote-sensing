#!/bin/bash
#SBATCH --job-name=mf-score-eval
#SBATCH --account=def-juliana2
#SBATCH --output=logs/mf_score_eval_%A_%a.out
#SBATCH --error=logs/mf_score_eval_%A_%a.err
#SBATCH --array=0-3%2
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:h100:1

set -euo pipefail

REPO_ROOT="/project/6002520/yunjung1/MethaneFuse"
VENV_PATH="/project/6002520/yunjung1/venvs/methanefuse"

cd "$REPO_ROOT"
mkdir -p logs results/two_negative_group_cv

source "$VENV_PATH/bin/activate"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

case "$SLURM_ARRAY_TASK_ID" in
  0)
    RUN_SLUG="fold_1_baseline"
    ;;
  1)
    RUN_SLUG="fold_1_augmented"
    ;;
  2)
    RUN_SLUG="fold_2_baseline"
    ;;
  3)
    RUN_SLUG="fold_2_augmented"
    ;;
  *)
    echo "Invalid task ID: $SLURM_ARRAY_TASK_ID" >&2
    exit 2
    ;;
esac

TEST_CSV="$REPO_ROOT/data/custom/two_negative_group_cv/$RUN_SLUG/test.csv"

CHECKPOINT="$REPO_ROOT/checkpoints/two_negative_group_cv/$RUN_SLUG/train__val__head/ckpt_best_test.pth"

WEIGHTS="$REPO_ROOT/weights/panopticon_vitb14_teacher.pth"

OUTPUT_JSON="$REPO_ROOT/results/two_negative_group_cv/${RUN_SLUG}_heldout_with_scores.json"

for file in \
  "$TEST_CSV" \
  "$CHECKPOINT" \
  "$WEIGHTS"
do
  if [[ ! -f "$file" ]]; then
    echo "Missing required file: $file" >&2
    exit 3
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

echo "Run: $RUN_SLUG"
echo "Test: $TEST_CSV"
echo "Checkpoint: $CHECKPOINT"

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

echo "Completed: $RUN_SLUG"
echo "Result: $OUTPUT_JSON"
