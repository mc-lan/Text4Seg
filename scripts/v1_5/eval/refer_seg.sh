#!/bin/bash

# gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
gpu_list="0,1,2,3,4,5,6,7"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

SPLIT="refcocog|umd|val"    # refcoco + | unc | testA, testB, val,   refcocog | umd | test, val

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_refer_seg \
        --model-path ./checkpoints/llava-v1.5-7b-p16/ \
        --image-folder ./playground/data/refer_seg/ \
        --save_file ./llava/eval/ref_seg_results/ \
        --dataset_split $SPLIT \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

python llava/eval/eval_refer_seg.py --dataset_split $SPLIT --save_file ./llava/eval/ref_seg_results/llava-v1.5-7b-p16/
python llava/eval/eval_refer_comprehen.py --dataset_split $SPLIT --save_file ./llava/eval/ref_seg_results/llava-v1.5-7b-p16/
