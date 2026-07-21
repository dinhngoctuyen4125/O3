#!/bin/bash
#SBATCH --job-name=Tuyen
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=120G
#SBATCH --gres=gpu:2
#SBATCH --output=/home/user03/tuyen/O3/ScienceQA_expriment/logs/%x_%j.out
#SBATCH --error=/home/user03/tuyen/O3/ScienceQA_expriment/logs/%x_%j.err

echo "===== HOST ====="
hostname

echo "===== CUDA_VISIBLE_DEVICES ====="
echo "$CUDA_VISIBLE_DEVICES"

echo "===== GPU ====="
which nvidia-smi
nvidia-smi

echo "===== PYTORCH ====="
python -c "import torch; print('Torch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"

echo "===== DEVICES ====="
ls -l /dev/nvidia*