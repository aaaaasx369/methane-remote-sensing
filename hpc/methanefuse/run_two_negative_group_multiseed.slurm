#!/bin/bash
#SBATCH --job-name=mf-multiseed
#SBATCH --account=def-juliana2
#SBATCH --output=logs/mf_multiseed_%A_%a.out
#SBATCH --error=logs/mf_multiseed_%A_%a.err
#SBATCH --array=0-11%2
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --exclude=fc10512

set -euo pipefail

REPO_ROOT="/project/6002520/yunjung1/MethaneFuse"
DATA_ROOT="$REPO_ROOT/data/custom/two_negative_group_cv"
VENV_PATH="/project/6002520/yunjung1/venvs/methanefuse"

CHECKPOINT_ROOT="$REPO_ROOT/checkpoints/two_negative_group_multiseed"
RESULT_ROOT="$REPO_ROOT/results/two_negative_group_multiseed"

INITIAL_CHECKPOINT="$REPO_ROOT/checkpoints/classification/methanefuse_cls_480m.pt"
PANOPTICON_WEIGHTS="$REPO_ROOT/weights/panopticon_vitb14_teacher.pth"

cd "$REPO_ROOT"

mkdir -p \
  logs \
  "$CHECKPOINT_ROOT" \
  "$RESULT_ROOT"

source "$VENV_PATH/bin/activate"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# --------------------------------------------------
# 12 tasks =
# 4 experiment settings x 3 random seeds
# --------------------------------------------------

RUNS=(
  "fold_1_baseline"
  "fold_1_augmented"
  "fold_2_baseline"
  "fold_2_augmented"
)

SEEDS=(
  42
  123
  2026
)

TASK_ID="$SLURM_ARRAY_TASK_ID"

RUN_INDEX=$(( TASK_ID / 3 ))
SEED_INDEX=$(( TASK_ID % 3 ))

RUN_SLUG="${RUNS[$RUN_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"

TRAIN_CSV="$DATA_ROOT/$RUN_SLUG/train.csv"
VAL_CSV="$DATA_ROOT/$RUN_SLUG/val.csv"
TEST_CSV="$DATA_ROOT/$RUN_SLUG/test.csv"

RUN_CHECKPOINT_DIR="$CHECKPOINT_ROOT/$RUN_SLUG/seed_$SEED"

VAL_OUTPUT="$RESULT_ROOT/${RUN_SLUG}_seed_${SEED}_val.json"
TEST_OUTPUT="$RESULT_ROOT/${RUN_SLUG}_seed_${SEED}_heldout.json"

mkdir -p "$RUN_CHECKPOINT_DIR"

echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Run: $RUN_SLUG"
echo "Seed: $SEED"
echo "Node: $(hostname)"
echo "Python: $(which python)"
echo "=================================================="

nvidia-smi

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable.")

free_bytes, total_bytes = torch.cuda.mem_get_info()

print("GPU:", torch.cuda.get_device_name(0))
print("Free GPU memory GB:", free_bytes / 1024**3)
print("Total GPU memory GB:", total_bytes / 1024**3)
PY

for file in \
  "$TRAIN_CSV" \
  "$VAL_CSV" \
  "$TEST_CSV" \
  "$INITIAL_CHECKPOINT" \
  "$PANOPTICON_WEIGHTS"
do
  if [[ ! -f "$file" ]]; then
    echo "Missing required file: $file" >&2
    exit 2
  fi
done

echo
echo "Starting training..."
echo "Train: $TRAIN_CSV"
echo "Validation: $VAL_CSV"
echo "Seed: $SEED"

TRAIN_CSV="$TRAIN_CSV" \
TEST_CSV="$VAL_CSV" \
WEIGHTS="$PANOPTICON_WEIGHTS" \
STAGE_A_CHECKPOINT="$INITIAL_CHECKPOINT" \
CHECKPOINT_DIR="$RUN_CHECKPOINT_DIR" \
DEVICE="cuda" \
BATCH_SIZE="4" \
EPOCHS="15" \
LORA_RANK="4" \
LORA_ALPHA="8" \
BACKBONE_LR="1e-5" \
HEAD_LR="1e-4" \
WEIGHT_DECAY="1e-3" \
NUM_WORKERS="4" \
ROW_FUSION_MODE="max" \
USE_WANDB=0 \
bash scripts/train/run_query.sh \
  --seed "$SEED" \
  --warmup_steps 20 \
  --lr_scheduler none

BEST_CHECKPOINT="$RUN_CHECKPOINT_DIR/train__val__head/ckpt_best_test.pth"

if [[ ! -f "$BEST_CHECKPOINT" ]]; then
  echo "Missing checkpoint: $BEST_CHECKPOINT" >&2
  exit 3
fi

echo
echo "Training completed."
echo "Checkpoint: $BEST_CHECKPOINT"

# --------------------------------------------------
# Validation scores
# --------------------------------------------------

echo
echo "Evaluating validation set..."

python scripts/eval/evaluate_classification.py \
  --eval_csv "$VAL_CSV" \
  --checkpoint "$BEST_CHECKPOINT" \
  --weights "$PANOPTICON_WEIGHTS" \
  --stage b \
  --lora_rank 4 \
  --lora_alpha 8 \
  --batch_size 4 \
  --num_workers 4 \
  --device cuda \
  --row_fusion_mode max \
  --output_json "$VAL_OUTPUT"

# --------------------------------------------------
# Held-out test scores
# --------------------------------------------------

echo
echo "Evaluating held-out test set..."

python scripts/eval/evaluate_classification.py \
  --eval_csv "$TEST_CSV" \
  --checkpoint "$BEST_CHECKPOINT" \
  --weights "$PANOPTICON_WEIGHTS" \
  --stage b \
  --lora_rank 4 \
  --lora_alpha 8 \
  --batch_size 4 \
  --num_workers 4 \
  --device cuda \
  --row_fusion_mode max \
  --output_json "$TEST_OUTPUT"

echo
echo "=================================================="
echo "COMPLETED"
echo "Run: $RUN_SLUG"
echo "Seed: $SEED"
echo "Validation: $VAL_OUTPUT"
echo "Held-out: $TEST_OUTPUT"
echo "=================================================="
