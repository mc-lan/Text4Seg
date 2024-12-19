# Experimental environment: 4 * A100
# 80GB GPU memory
# Note: TorchAcc is currently only available internally.

export USE_TORCHACC=1
export XLA_EXPERIMENTAL=nonzero:masked_select

export XLA_PERSISTENT_CACHE_PATH=./output/compiled_cache/qwen-72b-chat
mkdir -p $XLA_PERSISTENT_CACHE_PATH

NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model_type qwen-72b-chat \
    --model_layer_cls_name QWenBlock \
    --dataset codefuse-python-en \
    --sft_type lora \
    --output_dir output_qwen_72b \
    --num_train_epochs 1 \
    --max_length 2048 \
    --batch_size 8 \
    --use_flash_attn true \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing no \
    --tuner_backend 'peft' \
    --eval_steps 200 \
    --save_steps 200 \
    --logging_steps 100 \
    --acc_steps 100 \
    --metric_warmup_step 0.1 \
    --report_to 'none' \
    --fsdp_num 4 \
