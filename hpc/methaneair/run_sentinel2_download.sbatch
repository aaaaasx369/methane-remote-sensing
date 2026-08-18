#!/bin/bash
#SBATCH --account=def-juliana2
#SBATCH --job-name=mair_s2_dl
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/mair_s2_dl_%j.out
#SBATCH --error=logs/mair_s2_dl_%j.err

set -euo pipefail

PROJECT=/project/6002520/yunjung1/MethaneFuse
cd "$PROJECT"
mkdir -p logs

source /project/6002520/yunjung1/venvs/carbonmapper311/bin/activate

python methaneair_full_pipeline/build_methaneair_ground_truth.py \
  --project-root "$PROJECT"

python methaneair_full_pipeline/download_sentinel2_matches.py \
  --project-root "$PROJECT" \
  --include-candidate-negatives \
  --resume
