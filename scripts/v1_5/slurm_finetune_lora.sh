#!/bin/bash

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29000
NNODES=2
WORKER_GPU=16
NODE_RANK=$SLURM_PROCID

accelerate launch --config_file "scripts/default_config.yaml" \
    --num_processes $WORKER_GPU \
    --num_machines $NNODES \
    --machine_rank $NODE_RANK \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    llava/train/train_xformers.py \
    --lora_enable True --lora_r 64 --lora_alpha 128 --mm_projector_lr 2e-5 \
    --model_name_or_path ./pre_trained/vicuna-7b-v1.5/ \
    --version v1 \
    --data_path ./playground/data/ \
    --image_folder ./playground/data/ \
    --vision_tower ./pre_trained/clip-vit-large-patch14-336/ \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio anyres \
    --group_by_modality_length True \
    --bf16 False \
    --fp16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-lora-r64-p24 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard