#!/bin/bash
#SBATCH --job-name=check_env
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:2
#SBATCH --output=/home/user03/tuyen/O3/ScienceQA_expriment/logs/%x_%j.out
#SBATCH --error=/home/user03/tuyen/O3/ScienceQA_expriment/logs/%x_%j.err

echo "================ HOST ================"
hostname

echo
echo "================ GPU ================"
nvidia-smi

echo
echo "================ CUDA_VISIBLE_DEVICES ================"
echo "$CUDA_VISIBLE_DEVICES"

echo
echo "================ CONDA ================"
echo "CONDA_PREFIX = $CONDA_PREFIX"
echo "CONDA_DEFAULT_ENV = $CONDA_DEFAULT_ENV"

echo
echo "================ PYTHON ================"
which python
python --version

echo
echo "================ TORCH INFO ================"
python - <<'EOF'
import torch

print("torch file            =", torch.__file__)
print("torch version         =", torch.__version__)
print("torch.version.cuda    =", torch.version.cuda)
print("cuda.is_available     =", torch.cuda.is_available())
print("cuda.device_count     =", torch.cuda.device_count())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
EOF

echo
echo "================ CUDA TEST ================"
python - <<'EOF'
import torch

try:
    x = torch.randn(2, 2, device="cuda")
    y = torch.randn(2, 2, device="cuda")
    print("CUDA matmul OK")
    print(x @ y)
except Exception as e:
    print("CUDA test failed:")
    print(e)
EOF

echo
echo "================ PIP PACKAGES ================"
pip show torch torchvision torchaudio transformers peft accelerate deepspeed

echo
echo "================ ENV ================"
env | egrep 'CUDA|NCCL|LD_LIBRARY_PATH|PATH|CONDA'