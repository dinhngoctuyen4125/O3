#!/bin/bash
#SBATCH --output=/home/user03/tuyen/O3/ScienceQA_expriment/logs/%x_%j.out
#SBATCH --error=/home/user03/tuyen/O3/ScienceQA_expriment/logs/%x_%j.err

echo "===== HOST ====="
hostname

echo "===== GPU ====="
nvidia-smi

echo "===== PYTORCH ====="

python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"