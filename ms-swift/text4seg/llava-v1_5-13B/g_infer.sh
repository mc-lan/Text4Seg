#!/bin/bash

# gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
gpu_list="0,1,2,3,4,5,6,7"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL_TYPE="llava1_5-13b-instruct"
MODEL_BASE_PATH="checkpoints/grefcoco_2e_lr2e-4-bs128_r64_16/llava1_5-13b-instruct/v0-20240923-140120/checkpoint-6478"
MODEL_SUFFIX="-merged"
MODEL_PATH="${MODEL_BASE_PATH}${MODEL_SUFFIX}"
SAVE_FILE="output_eval/grefcoco_2e_lr2e-4-bs128_r64_16/"


if [ ! -d "$MODEL_PATH" ]; then
    echo "Model path does not exist. Running swift export..."
    CUDA_VISIBLE_DEVICES=0 swift export \
        --ckpt_dir $MODEL_BASE_PATH \
        --model_id_or_path "checkpoints/refcoco_5e_lr2e-4-bs128_r64_16/llava1_5-13b-instruct/v2-20240920-115927/checkpoint-33925-merged" \
        --merge_lora true
fi 

# SPLIT_OPTIONS=("grefcoco|unc|testA" "grefcoco|unc|testB" "grefcoco|unc|val")
SPLIT_OPTIONS="grefcoco|unc|val"

CUDA_VISIBLE_DEVICES=0,1 python -m text4seg.infer_refer_seg \
            --model_type $MODEL_TYPE \
            --model_id_or_path $MODEL_PATH \
            --dataset_split $SPLIT_OPTIONS \
            --save_file $SAVE_FILE \
            --visual_tokens 16 \
            --num-chunks 8 \
            --chunk-idx 4 &

CUDA_VISIBLE_DEVICES=2,3 python -m text4seg.infer_refer_seg \
            --model_type $MODEL_TYPE \
            --model_id_or_path $MODEL_PATH \
            --dataset_split $SPLIT_OPTIONS \
            --save_file $SAVE_FILE \
            --visual_tokens 16 \
            --num-chunks 8 \
            --chunk-idx 5 &

CUDA_VISIBLE_DEVICES=4,5 python -m text4seg.infer_refer_seg \
            --model_type $MODEL_TYPE \
            --model_id_or_path $MODEL_PATH \
            --dataset_split $SPLIT_OPTIONS \
            --save_file $SAVE_FILE \
            --visual_tokens 16 \
            --num-chunks 8 \
            --chunk-idx 6 &

CUDA_VISIBLE_DEVICES=6,7 python -m text4seg.infer_refer_seg \
            --model_type $MODEL_TYPE \
            --model_id_or_path $MODEL_PATH \
            --dataset_split $SPLIT_OPTIONS \
            --save_file $SAVE_FILE \
            --visual_tokens 16 \
            --num-chunks 8 \
            --chunk-idx 7 &

wait

python -m text4seg.eval_refer_seg --dataset_split  $SPLIT_OPTIONS --save_file ${SAVE_FILE}${MODEL_TYPE}


