#!/bin/bash

# gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
gpu_list="0,1,2,3,4,5,6,7"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL_TYPE="llava1_5-13b-instruct"
MODEL_BASE_PATH="checkpoints/refcoco_5e_lr2e-4-bs128_r64_16/llava1_5-13b-instruct/v2-20240920-115927/checkpoint-33925"
MODEL_SUFFIX="-merged"
MODEL_PATH="${MODEL_BASE_PATH}${MODEL_SUFFIX}"
SAVE_FILE="output_eval/refcoco_5e_lr2e-4-bs128_r64_16/"


if [ ! -d "$MODEL_PATH" ]; then
    echo "Model path does not exist. Running swift export..."
    CUDA_VISIBLE_DEVICES=0 swift export \
        --ckpt_dir $MODEL_BASE_PATH \
        --model_id_or_path "checkpoints/llava-1_5-13b-hf" \
        --merge_lora true
fi 

# SPLIT_OPTIONS=("refcoco|unc|val" "refcoco|unc|testA" "refcoco|unc|testB" "refcoco+|unc|val" "refcoco+|unc|testA" "refcoco+|unc|testB" "refcocog|umd|val" "refcocog|umd|test")
# SPLIT_OPTIONS=("refcoco|unc|val" "refcoco|unc|testA" "refcoco|unc|testB" "refcoco+|unc|val")
# SPLIT_OPTIONS=("refcoco+|unc|testA" "refcoco+|unc|testB" "refcocog|umd|val" "refcocog|umd|test")

# SPLIT_OPTIONS=("refcoco|unc|val" "refcoco|unc|testA")
# SPLIT_OPTIONS=("refcoco|unc|testB" "refcoco+|unc|val")
# SPLIT_OPTIONS=("refcoco+|unc|testA" "refcocog|umd|val")
# SPLIT_OPTIONS=("refcoco+|unc|testB" "refcocog|umd|test")

SPLIT_OPTIONS="refcocog|umd|test"

CUDA_VISIBLE_DEVICES=0,1 python -m my_scripts.infer_refer_seg \
            --model_type $MODEL_TYPE \
            --model_id_or_path $MODEL_PATH \
            --dataset_split $SPLIT_OPTIONS \
            --save_file $SAVE_FILE \
            --visual_tokens 16 \
            --num-chunks 4 \
            --chunk-idx 0 &

CUDA_VISIBLE_DEVICES=2,3 python -m my_scripts.infer_refer_seg \
            --model_type $MODEL_TYPE \
            --model_id_or_path $MODEL_PATH \
            --dataset_split $SPLIT_OPTIONS \
            --save_file $SAVE_FILE \
            --visual_tokens 16 \
            --num-chunks 4 \
            --chunk-idx 1 &

CUDA_VISIBLE_DEVICES=4,5 python -m my_scripts.infer_refer_seg \
            --model_type $MODEL_TYPE \
            --model_id_or_path $MODEL_PATH \
            --dataset_split $SPLIT_OPTIONS \
            --save_file $SAVE_FILE \
            --visual_tokens 16 \
            --num-chunks 4 \
            --chunk-idx 2 &

CUDA_VISIBLE_DEVICES=6,7 python -m my_scripts.infer_refer_seg \
            --model_type $MODEL_TYPE \
            --model_id_or_path $MODEL_PATH \
            --dataset_split $SPLIT_OPTIONS \
            --save_file $SAVE_FILE \
            --visual_tokens 16 \
            --num-chunks 4 \
            --chunk-idx 3 &

wait

python -m my_scripts.eval_refer_seg --dataset_split $SPLIT_OPTIONS --save_file ${SAVE_FILE}${MODEL_TYPE}

python -m my_scripts.eval_refer_comprehen --dataset_split $SPLIT_OPTIONS --save_file ${SAVE_FILE}${MODEL_TYPE}

