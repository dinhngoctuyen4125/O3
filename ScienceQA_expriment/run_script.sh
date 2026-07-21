#!/bin/bash
#SBATCH --job-name=Tuyen
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30          # khoảng 10 CPU/GPU
#SBATCH --mem=120G                  # có thể điều chỉnh
#SBATCH --gres=gpu:3
#SBATCH --output=/home/user03/tuyen/O3/ScienceQA_expriment/logs/%x_%j.out
#SBATCH --error=/home/user03/tuyen/O3/ScienceQA_expriment/logs/%x_%j.err

cd /home/user03/tuyen/O3/ScienceQA_expriment

torchrun \
    --nproc_per_node=3 \
    --nnodes=1 \
    train_basemodel.py \
    --model_name_or_path NousResearch/Llama-2-7b-hf \
    --data_path ./data/qa_all/qaall_train.json \
    --bf16 True \
    --output_dir O3_LLAMA2_ScienceQA \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --deepspeed ./configs/default_offload_opt_param.json \
    --tf32 True