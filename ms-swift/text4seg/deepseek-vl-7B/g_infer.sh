#!/bin/bash

# gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
gpu_list="0,1,2,3,4,5,6,7"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL_TYPE="deepseek-vl-7b-chat"
MODEL_BASE_PATH="checkpoints/grefcoco_2e_lr2e-4-bs128_r64_16/deepseek-vl-7b-chat/v0-20240918-020541/checkpoint-6476"
MODEL_SUFFIX="-merged"
MODEL_PATH="${MODEL_BASE_PATH}${MODEL_SUFFIX}"
SAVE_FILE="output_eval/grefcoco_2e_lr2e-4-bs128_r64_16/"


if [ ! -d "$MODEL_PATH" ]; then
    echo "Model path does not exist. Running swift export..."
    CUDA_VISIBLE_DEVICES=0 swift export \
        --ckpt_dir $MODEL_BASE_PATH \
        --model_id_or_path "checkpoints/refcoco_5e_lr2e-4-bs128_r64_16/deepseek-vl-7b-chat/v1-20240917-165420/checkpoint-33925-merged" \
        --merge_lora true
fi

# SPLIT_OPTIONS=("grefcoco|unc|testA" "grefcoco|unc|testB" "grefcoco|unc|val")
SPLIT_OPTIONS=("grefcoco|unc|val")


for SPLIT in "${SPLIT_OPTIONS[@]}"; do
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m text4seg.infer_refer_seg \
            --model_type $MODEL_TYPE \
            --model_id_or_path $MODEL_PATH \
            --dataset_split $SPLIT \
            --save_file $SAVE_FILE \
            --visual_tokens 16 \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX &
    done

    wait
    
    python -m text4seg.eval_refer_seg --dataset_split $SPLIT --save_file ${SAVE_FILE}${MODEL_TYPE}
done
