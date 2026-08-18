#!/bin/bash
#SBATCH --account=def-juliana2
#SBATCH --job-name=mair_s2_v2full
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/mair_s2_v2full_%j.out
#SBATCH --error=logs/mair_s2_v2full_%j.err

set -euo pipefail

PROJECT=/project/6002520/yunjung1/MethaneFuse

cd "$PROJECT"
mkdir -p logs

source /project/6002520/yunjung1/venvs/carbonmapper311/bin/activate

python methaneair_full_pipeline/download_sentinel2_matches_best_qa_v2.py \
  --project-root "$PROJECT" \
  --ground-truth \
  data/methaneair_full/ground_truth_confirmed_search_ready_v2.csv \
  --output-subdir sentinel2_best_qa_v2_full \
  --max-candidates-per-slot 10
