#!/bin/bash
#SBATCH --account=def-juliana2
#SBATCH --job-name=mf_s2_506eval
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=h100:1
#SBATCH --exclude=fc10512
#SBATCH --output=logs/mf_s2_506eval_%j.out
#SBATCH --error=logs/mf_s2_506eval_%j.err

set -euo pipefail

cd /project/6002520/yunjung1/MethaneFuse

module purge
module load python/3.11
module load scipy-stack/2025a

source /project/6002520/yunjung1/venvs/methanefuse/bin/activate

mkdir -p results/eval/methaneair_full_strict506

echo "===== START STRICT506 ZERO-SHOT ====="
date

python scripts/eval/evaluate_classification_with_predictions.py \
  --eval_csv data/custom/methaneair_full_strict506_480m_eval.csv \
  --checkpoint checkpoints/classification/methanefuse_cls_480m.pt \
  --stage b \
  --batch_size 4 \
  --num_workers 0 \
  --row_fusion_mode max \
  --output_json results/eval/methaneair_full_strict506/metrics.json \
  --output_predictions_csv results/eval/methaneair_full_strict506/predictions.csv

echo
echo "===== METRICS ====="
cat results/eval/methaneair_full_strict506/metrics.json

date
echo "===== EVALUATION COMPLETE ====="
