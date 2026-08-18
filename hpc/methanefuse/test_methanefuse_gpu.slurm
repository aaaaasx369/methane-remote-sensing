#!/bin/bash
#SBATCH --account=def-juliana2
#SBATCH --job-name=mf-gpu-test
#SBATCH --output=logs/mf_gpu_test_%j.out
#SBATCH --error=logs/mf_gpu_test_%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gpus=h100:1

set -euo pipefail

module --force purge
module load StdEnv/2023
module load gcc/12.3
module load python/3.10.13
module load arrow/18.1.0
module load cuda/12.6

source /project/def-juliana2/yunjung1/venvs/methanefuse/bin/activate

cd /project/def-juliana2/yunjung1/MethaneFuse

echo "Host: $(hostname)"
echo "Python: $(which python)"

nvidia-smi

python - <<'PY'
import torch
import pyarrow
import src.data.multisensor

print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("pyarrow:", pyarrow.__version__)
print("multisensor:", src.data.multisensor.__file__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("Allocated GPU is not visible to PyTorch.")

print("GPU:", torch.cuda.get_device_name(0))

x = torch.randn(2048, 2048, device="cuda")
y = x @ x

torch.cuda.synchronize()

print("Result:", y.shape)
print("Device:", y.device)
print("GPU smoke test passed.")
PY
