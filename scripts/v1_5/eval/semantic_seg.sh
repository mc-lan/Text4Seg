#!/bin/bash

# gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
gpu_list="0,1,2,3,4,5,6,7"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

# PAS20 ADE20K PC59
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_semantic_seg \
        --model-path checkpoints/llava-v1.5-7b-p24 \
        --save_file ./llava/eval/semantic_seg_results/ \
        --dataset_split "PAS20" \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

python -m llava.eval.eval_semantic_seg --dataset_split "PAS20" --save_file ./llava/eval/semantic_seg_results/llava-v1.5-7b-p24