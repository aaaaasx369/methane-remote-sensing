#!/bin/bash
#SBATCH --account=def-juliana2
#SBATCH --job-name=mf_mair_360
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=h100:1
#SBATCH --output=logs/mf_mair_360_%j.out
#SBATCH --error=logs/mf_mair_360_%j.err

set -euo pipefail

cd /project/def-juliana2/yunjung1/MethaneFuse

module purge
module load python/3.11
module load scipy-stack/2025a

VENV_LINE="$(
  grep -m1 -E \
  '^[[:space:]]*(export[[:space:]]+)?VENV=' \
  run_exact3_methanefuse.sh
)"

eval "$VENV_LINE"
source "$VENV/bin/activate"
hash -r

mkdir -p results/eval/methaneair_s2_p1_360m

python scripts/eval/evaluate_classification_with_predictions.py \
  --eval_csv data/custom/methaneair_s2_p1_zero_shot_eval_final16.csv \
  --checkpoint checkpoints/classification/methanefuse_cls_360m.pt \
  --stage b \
  --batch_size 4 \
  --num_workers 0 \
  --row_fusion_mode max \
  --output_json results/eval/methaneair_s2_p1_360m/final16_360m_with_predictions.json \
  --output_predictions_csv results/eval/methaneair_s2_p1_360m/final16_360m_predictions.csv
