#!/bin/bash

# Experimental environment: 8 * A800
# 8 * 40GB GPU memory

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-6000}
NNODES=4
WORKER_GPU=32
NODE_RANK=${RANK:-0}

accelerate launch --config_file "my_scripts/llava-v1_5-13B/config.yaml" \
    --num_processes $WORKER_GPU \
    --num_machines $NNODES \
    --machine_rank $NODE_RANK \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    examples/pytorch/llm/llm_sft.py \
    --model_type llava1_5-13b-instruct \
    --model_id_or_path "" \
    --model_revision master \
    --sft_type lora \
    --tuner_backend peft \
    --template_type AUTO \
    --output_dir checkpoints/grefer_2e_lr2e-4_bs128_r64_16 \
    --ddp_backend nccl \
    --dataset "datasets/json_files/grefcoco_16_2_llava.json" \
    --train_dataset_sample -1 \
    --num_train_epochs 2 \
    --max_length 2048 \
    --check_dataset_strategy warning \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout_p 0.05 \
    --lora_target_modules ALL \
    --gradient_checkpointing true \
    --batch_size 4 \
    --lr_scheduler_type 'cosine' \
    --weight_decay 0.0 \
    --learning_rate 2e-4 \
    --warmup_ratio 0.03 \
    --eval_steps 100000 \
    --save_steps 500 \
    --save_total_limit 1 \
    --logging_steps 1 \
    --use_flash_attn false \