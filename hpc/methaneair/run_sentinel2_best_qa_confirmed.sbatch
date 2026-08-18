#!/bin/bash
#SBATCH --account=def-juliana2
#SBATCH --job-name=mair_s2_bestqa
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/mair_s2_bestqa_%j.out
#SBATCH --error=logs/mair_s2_bestqa_%j.err

set -euo pipefail

PROJECT=/project/6002520/yunjung1/MethaneFuse

cd "$PROJECT"
mkdir -p logs

source /project/6002520/yunjung1/venvs/carbonmapper311/bin/activate

python methaneair_full_pipeline/download_sentinel2_matches_best_qa.py \
  --project-root "$PROJECT" \
  --ground-truth \
  data/methaneair_full/ground_truth_confirmed_search_ready.csv \
  --max-candidates-per-slot 10 \
  --resume
