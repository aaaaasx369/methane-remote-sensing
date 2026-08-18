#!/bin/bash
#SBATCH --account=def-juliana2
#SBATCH --job-name=mf_exact_expand
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=h100:1
#SBATCH --exclude=fc10512
#SBATCH --output=logs/mf_exact_expand_%j.out
#SBATCH --error=logs/mf_exact_expand_%j.err

set -euo pipefail

cd /project/6002520/yunjung1/MethaneFuse

module purge
module load python/3.11
module load scipy-stack/2025a

source /project/6002520/yunjung1/venvs/methanefuse/bin/activate

mkdir -p results/eval/newer_than_final16/exact_s2_expanded
mkdir -p logs

echo "===== EXPANDED 12 ====="

python scripts/eval/evaluate_classification_with_predictions.py \
  --eval_csv data/custom/exact_s2_external_eval_expanded12.csv \
  --checkpoint checkpoints/classification/methanefuse_cls_480m.pt \
  --stage b \
  --batch_size 4 \
  --num_workers 0 \
  --row_fusion_mode max \
  --output_json results/eval/newer_than_final16/exact_s2_expanded/expanded12.json \
  --output_predictions_csv results/eval/newer_than_final16/exact_s2_expanded/expanded12_predictions.csv

echo
echo "===== STRICT 7 ====="

python scripts/eval/evaluate_classification_with_predictions.py \
  --eval_csv data/custom/exact_s2_external_eval_strict7.csv \
  --checkpoint checkpoints/classification/methanefuse_cls_480m.pt \
  --stage b \
  --batch_size 4 \
  --num_workers 0 \
  --row_fusion_mode max \
  --output_json results/eval/newer_than_final16/exact_s2_expanded/strict7.json \
  --output_predictions_csv results/eval/newer_than_final16/exact_s2_expanded/strict7_predictions.csv
