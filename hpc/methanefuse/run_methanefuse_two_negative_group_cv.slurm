#!/bin/bash
#SBATCH --job-name=mf-2neg-cv
#SBATCH --account=def-juliana2
#SBATCH --output=logs/mf_2neg_%A_%a.out
#SBATCH --error=logs/mf_2neg_%A_%a.err
#SBATCH --array=0-3%2
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1

set -euo pipefail

REPO_ROOT="/project/6002520/yunjung1/MethaneFuse"
DATA_ROOT="$REPO_ROOT/data/custom/two_negative_group_cv"
VENV_PATH="/project/6002520/yunjung1/venvs/methanefuse"

INITIAL_CHECKPOINT="$REPO_ROOT/checkpoints/classification/methanefuse_cls_480m.pt"
PANOPTICON_WEIGHTS="$REPO_ROOT/weights/panopticon_vitb14_teacher.pth"

RESULT_ROOT="$REPO_ROOT/results/two_negative_group_cv"
CHECKPOINT_ROOT="$REPO_ROOT/checkpoints/two_negative_group_cv"

cd "$REPO_ROOT"

mkdir -p \
  logs \
  "$RESULT_ROOT" \
  "$CHECKPOINT_ROOT"

source "$VENV_PATH/bin/activate"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

echo "=================================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
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
    raise RuntimeError("CUDA is not available on the compute node.")

print("GPU:", torch.cuda.get_device_name(0))
PY

for required_file in \
  "$DATA_ROOT/folds.tsv" \
  "$INITIAL_CHECKPOINT" \
  "$PANOPTICON_WEIGHTS"
do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing required file: $required_file" >&2
    exit 2
  fi
done

RUN_LINE=$(
  awk -F '\t' \
    -v target="$SLURM_ARRAY_TASK_ID" \
    '$1 == target {print; exit}' \
    "$DATA_ROOT/folds.tsv"
)

if [[ -z "$RUN_LINE" ]]; then
  echo \
    "No run record for array index ${SLURM_ARRAY_TASK_ID}" \
    >&2
  exit 3
fi

IFS=$'\t' read -r \
  RUN_INDEX \
  RUN_SLUG \
  HELD_OUT_SITE \
  TRAIN_REL \
  VAL_REL \
  TEST_REL \
  <<< "$RUN_LINE"

TRAIN_CSV="$DATA_ROOT/$TRAIN_REL"
VAL_CSV="$DATA_ROOT/$VAL_REL"
TEST_CSV="$DATA_ROOT/$TEST_REL"

echo "Run index: $RUN_INDEX"
echo "Run slug: $RUN_SLUG"
echo "Held-out site: $HELD_OUT_SITE"
echo "Train CSV: $TRAIN_CSV"
echo "Validation CSV: $VAL_CSV"
echo "Test CSV: $TEST_CSV"

RUNTIME_DIR="$SLURM_TMPDIR/manifests/$RUN_SLUG"
mkdir -p "$RUNTIME_DIR"

python - \
  "$TRAIN_CSV" \
  "$VAL_CSV" \
  "$TEST_CSV" \
  "$RUNTIME_DIR" <<'PY'
from pathlib import Path
import sys
import pandas as pd

sources = [
    Path(sys.argv[1]),
    Path(sys.argv[2]),
    Path(sys.argv[3]),
]

runtime_dir = Path(sys.argv[4])
runtime_dir.mkdir(parents=True, exist_ok=True)

required_columns = [
    "id",
    "label",
    "s2_0_path",
    "s2_90_path",
    "s2_360_path",
]

for source in sources:
    if not source.exists():
        raise FileNotFoundError(source)

    df = pd.read_csv(source, low_memory=False)

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{source}: missing columns {missing_columns}"
        )

    for column in [
        "s2_0_path",
        "s2_90_path",
        "s2_360_path",
    ]:
        df[column] = df[column].map(
            lambda value: str(
                Path(str(value)).expanduser().resolve()
            )
        )

        missing_files = [
            value
            for value in df[column]
            if not Path(value).exists()
        ]

        if missing_files:
            raise FileNotFoundError(
                f"{source.name}: {column} missing file, "
                f"first={missing_files[0]}"
            )

    destination = runtime_dir / source.name
    df.to_csv(destination, index=False)

    print(
        f"{source} -> {destination} "
        f"({len(df)} rows)"
    )
PY

TRAIN_RUNTIME="$RUNTIME_DIR/train.csv"
VAL_RUNTIME="$RUNTIME_DIR/val.csv"
TEST_RUNTIME="$RUNTIME_DIR/test.csv"

RUN_CHECKPOINT_DIR="$CHECKPOINT_ROOT/$RUN_SLUG"
mkdir -p "$RUN_CHECKPOINT_DIR"
RESULT_JSON="$RESULT_ROOT/${RUN_SLUG}_heldout.json"

echo "Starting Stage-B adaptation..."

TRAIN_CSV="$TRAIN_RUNTIME" \
TEST_CSV="$VAL_RUNTIME" \
WEIGHTS="$PANOPTICON_WEIGHTS" \
STAGE_A_CHECKPOINT="$INITIAL_CHECKPOINT" \
CHECKPOINT_DIR="$RUN_CHECKPOINT_DIR" \
RUN_NAME="$RUN_SLUG" \
DEVICE="cuda" \
BATCH_SIZE="${BATCH_SIZE:-4}" \
EPOCHS="${EPOCHS:-15}" \
LORA_RANK="${LORA_RANK:-4}" \
LORA_ALPHA="${LORA_ALPHA:-8}" \
BACKBONE_LR="${BACKBONE_LR:-1e-5}" \
HEAD_LR="${HEAD_LR:-1e-4}" \
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-3}" \
NUM_WORKERS="${SLURM_CPUS_PER_TASK:-4}" \
ROW_FUSION_MODE="max" \
USE_WANDB=0 \
bash scripts/train/run_query.sh \
  --warmup_steps 20 \
  --lr_scheduler none

BEST_CHECKPOINT="$RUN_CHECKPOINT_DIR/train__val__head/ckpt_best_test.pth"

if [[ ! -f "$BEST_CHECKPOINT" ]]; then
  echo "Missing best checkpoint: $BEST_CHECKPOINT" >&2
  exit 4
fi

echo "Evaluating held-out test set..."

python scripts/eval/evaluate_classification.py \
  --eval_csv "$TEST_RUNTIME" \
  --checkpoint "$BEST_CHECKPOINT" \
  --weights "$PANOPTICON_WEIGHTS" \
  --stage b \
  --lora_rank 4 \
  --lora_alpha 8 \
  --batch_size 4 \
  --num_workers "${SLURM_CPUS_PER_TASK:-4}" \
  --device cuda \
  --row_fusion_mode max \
  --output_json "$RESULT_JSON"

echo "=================================================="
echo "Completed: $RUN_SLUG"
echo "Held-out site: $HELD_OUT_SITE"
echo "Checkpoint: $BEST_CHECKPOINT"
echo "Result: $RESULT_JSON"
echo "=================================================="
