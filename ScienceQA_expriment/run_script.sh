#!/bin/bash
#SBATCH --job-name=Tuyen
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/user03/tuyen/O3/ScienceQA_expriment/logs/%x_%j.out
#SBATCH --error=/home/user03/tuyen/O3/ScienceQA_expriment/logs/%x_%j.err

echo "===== Host ====="
hostname

echo "===== GPU ====="
which nvidia-smi
nvidia-smi

echo "===== CUDA ====="
python -c "import torch; \
print('CUDA available:', torch.cuda.is_available()); \
print('CUDA version:', torch.version.cuda); \
print('GPU count:', torch.cuda.device_count()); \
print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"