#!/bin/bash

# Experimental environment: 8 * A800
# 8 * 40GB GPU memory
nproc_per_node=8

NPROC_PER_NODE=$nproc_per_node \
MASTER_PORT=29500 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model_type internvl2-8b \
    --model_id_or_path "" \
    --model_revision master \
    --sft_type lora \
    --tuner_backend peft \
    --template_type AUTO \
    --dtype fp16 \
    --output_dir checkpoints/grefer_2e_lr2e-4_bs128_r64_16 \
    --ddp_backend nccl \
    --dataset "datasets/json_files/grefcoco_16_2_internvl.json" \
    --train_dataset_sample -1 \
    --num_train_epochs 2 \
    --max_length 2048 \
    --check_dataset_strategy warning \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout_p 0.05 \
    --lora_target_modules DEFAULT \
    --gradient_checkpointing true \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --weight_decay 0.0 \
    --learning_rate 2e-4 \
    --lr_scheduler_type 'cosine' \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.03 \
    --eval_steps 100000 \
    --save_steps 500 \
    --save_total_limit 1 \
    --logging_steps 1 \
    --use_flash_attn false \
    --deepspeed 'default-zero2' \