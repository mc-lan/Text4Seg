#!/bin/bash

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29000
NNODES=2
WORKER_GPU=16
NODE_RANK=$SLURM_PROCID

accelerate launch --config_file "text4seg/llava-v1_5-13B/config.yaml" \
    --num_processes $WORKER_GPU \
    --num_machines $NNODES \
    --machine_rank $NODE_RANK \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    examples/pytorch/llm/llm_sft.py \
    --model_type internvl2-8b \
    --model_id_or_path "checkpoints/Intern-vl2-8b" \
    --model_revision master \
    --sft_type lora \
    --tuner_backend peft \
    --template_type AUTO \
    --output_dir checkpoints/refcoco_5e_lr2e-4_bs128_r64_16 \
    --ddp_backend nccl \
    --dataset "datasets/json_files/refcoco_16_2_internvl.json" \
    --train_dataset_sample -1 \
    --num_train_epochs 5 \
    --max_length 2048 \
    --check_dataset_strategy warning \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout_p 0.05 \
    --lora_target_modules DEFAULT \
    --gradient_checkpointing true \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type 'cosine' \
    --weight_decay 0.0 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.03 \
    --eval_steps 100000 \
    --save_steps 500 \
    --save_total_limit 1 \
    --logging_steps 1 \
    --use_flash_attn true \