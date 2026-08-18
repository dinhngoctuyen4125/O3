#!/bin/bash

#SBATCH --job-name=olora
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log
#SBATCH --partition=defq
#SBATCH --qos=short
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

BASE_MODEL="codellama/CodeLlama-7b-Instruct-hf"
for SCALE in 0.1
do
    for SEED in 0
    do
        OUTPUT_1="./SCALE_${SCALE}_seed_${SEED}_o_unlearn_lora_checkpoints/lora_forget"
        DATAPATH_1="../../Data-Collection/codellama/D_forget.json"

        # Train on D_forget.json
        python train_unlearn_lora_o.py \
            --base_model ${BASE_MODEL} \
            --data_path ${DATAPATH_1} \
            --output_dir ${OUTPUT_1} \
            --seed ${SEED} \
            --batch_size 128 \
            --micro_batch_size 8 \
            --num_epochs 15 \
            --learning_rate 3e-4 \
            --cutoff_len 256 \
            --val_set_size 1 \
            --lora_r 8 \
            --lora_alpha 16 \
            --lora_dropout 0.05 \
            --lora_weights "" \
            --ood_weight "1,1" \
            --orthogonal_loss_weight ${SCALE} \
            --train_on_inputs \
            --group_by_length \
            --add_eos_token

        # Eval on D_test.json
        # TESTPATH_1="./data/codellama/D_test.json"
        TESTPATH_1="../../Data-Collection/codellama/D_test.json"
        python eval_lora.py \
            --test_dataset ${TESTPATH_1} \
            --base_model ${BASE_MODEL} \
            --seed ${SEED} \
            --lora_weights ${OUTPUT_1}
    done
done