#!/bin/bash
#SBATCH --job-name=mf-5site-loso
#SBATCH --output=logs/mf_loso_%A_%a.out
#SBATCH --error=logs/mf_loso_%A_%a.err
#SBATCH --array=0-4
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/projects/MethaneFuse}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data/custom/five_site_loso}"
VENV_PATH="${VENV_PATH:-$HOME/venvs/methanefuse}"
INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-$REPO_ROOT/checkpoints/classification/methanefuse_cls_480m.pt}"
PANOPTICON_WEIGHTS="${PANOPTICON_WEIGHTS:-$REPO_ROOT/weights/panopticon_vitb14_teacher.pth}"
WV3_SRF="${WV3_SRF:-$REPO_ROOT/WV3_VNIR_SWIR_response.csv}"

cd "$REPO_ROOT"
mkdir -p logs results/five_site_loso checkpoints/five_site_loso
source "$VENV_PATH/bin/activate"

nvidia-smi || true
python --version

FOLD_LINE=$(awk -F '\t' -v target="$SLURM_ARRAY_TASK_ID" '$1 == target {print; exit}' "$DATA_ROOT/folds.tsv")
if [[ -z "$FOLD_LINE" ]]; then
  echo "No fold record for array index $SLURM_ARRAY_TASK_ID" >&2
  exit 2
fi

IFS=$'\t' read -r FOLD_INDEX FOLD_SLUG HELD_OUT_SITE TRAIN_REL VAL_REL TEST_REL <<< "$FOLD_LINE"
TRAIN_CSV="$DATA_ROOT/$TRAIN_REL"
VAL_CSV="$DATA_ROOT/$VAL_REL"
TEST_CSV="$DATA_ROOT/$TEST_REL"

RUNTIME_DIR="$SLURM_TMPDIR/manifests/$FOLD_SLUG"
mkdir -p "$RUNTIME_DIR"

python - "$DATA_ROOT" "$TRAIN_CSV" "$VAL_CSV" "$TEST_CSV" "$RUNTIME_DIR" <<'PY'
from pathlib import Path
import sys
import pandas as pd

data_root = Path(sys.argv[1]).resolve()
inputs = [Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])]
runtime_dir = Path(sys.argv[5])
runtime_dir.mkdir(parents=True, exist_ok=True)

for source in inputs:
    df = pd.read_csv(source, low_memory=False)
    for column in ["s2_0_path", "s2_90_path", "s2_360_path"]:
        def resolve(value):
            path = Path(str(value))
            if path.is_absolute():
                return str(path)
            return str((data_root / path).resolve())
        df[column] = df[column].apply(resolve)
        missing = [p for p in df[column] if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(
                f"{source.name}: {column} has missing files, first={missing[0]}"
            )
    destination = runtime_dir / source.name
    df.to_csv(destination, index=False)
    print(f"{source} -> {destination} ({len(df)} rows)")
PY

TRAIN_RUNTIME="$RUNTIME_DIR/train.csv"
VAL_RUNTIME="$RUNTIME_DIR/val.csv"
TEST_RUNTIME="$RUNTIME_DIR/test.csv"
RUN_NAME="$FOLD_SLUG"
CHECKPOINT_DIR="$REPO_ROOT/checkpoints/five_site_loso"
RESULT_JSON="$REPO_ROOT/results/five_site_loso/${FOLD_SLUG}_heldout.json"

TRAIN_CSV="$TRAIN_RUNTIME" \
TEST_CSV="$VAL_RUNTIME" \
WEIGHTS="$PANOPTICON_WEIGHTS" \
STAGE_A_CHECKPOINT="$INITIAL_CHECKPOINT" \
CHECKPOINT_DIR="$CHECKPOINT_DIR" \
RUN_NAME="$RUN_NAME" \
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
bash scripts/train/run_query.sh \
  --wv3_srf_csv "$WV3_SRF" \
  --warmup_steps 20 \
  --lr_scheduler none

BEST_CHECKPOINT="$CHECKPOINT_DIR/$RUN_NAME/ckpt_best_test.pth"
if [[ ! -f "$BEST_CHECKPOINT" ]]; then
  echo "Missing best checkpoint: $BEST_CHECKPOINT" >&2
  exit 3
fi

python scripts/eval/evaluate_classification.py \
  --eval_csv "$TEST_RUNTIME" \
  --checkpoint "$BEST_CHECKPOINT" \
  --weights "$PANOPTICON_WEIGHTS" \
  --stage b \
  --batch_size 4 \
  --num_workers "${SLURM_CPUS_PER_TASK:-4}" \
  --device cuda \
  --row_fusion_mode max \
  --wv3_srf_csv "$WV3_SRF" \
  --output_json "$RESULT_JSON"

echo "Completed fold $FOLD_INDEX, held-out site: $HELD_OUT_SITE"
echo "Result: $RESULT_JSON"
