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
echo "================ NVCC ================"
which nvcc || true
nvcc --version || true

echo
echo "================ NVIDIA DRIVER ================"
cat /proc/driver/nvidia/version || true

echo
echo "================ PYTHON ================"
which python
python --version

echo
echo "================ TORCH ================"
python - <<'EOF'
import torch

print("torch.__version__      =", torch.__version__)
print("torch.version.cuda     =", torch.version.cuda)
print("torch.cuda.is_available=", torch.cuda.is_available())
print("torch.cuda.device_count=", torch.cuda.device_count())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
EOF

echo
echo "================ TORCH BUILD ================"
pip show torch || true

echo
echo "================ DEEPSPEED ================"
python - <<'EOF'
try:
    import deepspeed
    print("deepspeed version:", deepspeed.__version__)
except Exception as e:
    print(e)
EOF

echo
echo "================ NCCL ================"
python - <<'EOF'
import torch
print("nccl available:", torch.distributed.is_nccl_available())
EOF

echo
echo "================ CUDA TEST ================"
python - <<'EOF'
import torch

try:
    x = torch.randn(2,2).cuda()
    y = torch.randn(2,2).cuda()
    print("CUDA matmul OK")
    print((x@y))
except Exception as e:
    print(e)
EOF

echo
echo "================ ENV ================"
env | egrep 'CUDA|NCCL|LD_LIBRARY_PATH|PATH'