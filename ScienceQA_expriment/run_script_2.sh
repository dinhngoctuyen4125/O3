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

python - <<EOF
import torch
import transformers
import peft
import accelerate

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("peft:", peft.__version__)
print("accelerate:", accelerate.__version__)
print("cuda:", torch.cuda.is_available())
EOF