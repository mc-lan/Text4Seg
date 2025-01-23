#!/bin/bash

# gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
gpu_list="0,1,2,3,4,5,6,7"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL_TYPE="qwen-vl-chat"
MODEL_BASE_PATH="./output_gref/list_grefer_2e_lr2e-4_bs128_r64_224/qwen-vl-chat/v0-20240915-011532/checkpoint-6476"
MODEL_SUFFIX="-merged"
MODEL_PATH="${MODEL_BASE_PATH}${MODEL_SUFFIX}"
SAVE_FILE="output_eval/list_refcoco_clef_5e_lr2e-4_bs128_r64_224/"


if [ ! -d "$MODEL_PATH" ]; then
    echo "Model path does not exist. Running swift export..."
    CUDA_VISIBLE_DEVICES=0 swift export \
        --ckpt_dir $MODEL_BASE_PATH \
        --model_id_or_path "output_new/list_refcoco_clef_5e_lr2e-4_bs128_r64_224/qwen-vl-chat/v3-20240912-152538/checkpoint-33925-merged" \
        --merge_lora true
fi

SPLIT_OPTIONS=("grefcoco|unc|testA" "grefcoco|unc|testB" "grefcoco|unc|val")
# SPLIT_OPTIONS=("grefcoco|unc|testA" "grefcoco|unc|val")


for SPLIT in "${SPLIT_OPTIONS[@]}"; do
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m my_scripts.infer_refer_seg \
            --model_type $MODEL_TYPE \
            --model_id_or_path $MODEL_PATH \
            --dataset_split $SPLIT \
            --save_file $SAVE_FILE \
            --visual_tokens 16 \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX &
    done

    wait
    
    python -m my_scripts.eval_refer_seg --dataset_split $SPLIT --save_file ${SAVE_FILE}${MODEL_TYPE}
done
