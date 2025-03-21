<div align="center">
<h1>Text4Seg: Reimagining Image Segmentation as Text Generation</h1>

<div>
    <a href='https://mc-lan.github.io/' target='_blank'>Mengcheng Lan</a><sup>1</sup>&emsp;
    <a href='https://chaofengc.github.io/' target='_blank'>Chaofeng Chen</a><sup>1</sup>&emsp;
    <a href='https://zytx121.github.io/' target='_blank'>Yue Zhou</a><sup>1</sup>&emsp;   
    <a href='https://angusmonroe.cn/' target='_blank'>Jiaxing Xu</a><sup>2</sup>&emsp;
    <a href='https://keyiping.wixsite.com/index' target='_blank'>Yiping Ke</a><sup>2</sup>&emsp;
    <a href='https://scholar.google.com.hk/citations?user=q4lnWaoAAAAJ&hl=en&inst=8669986779262753491&oi=ao' target='_blank'>Xinjiang Wang</a><sup>3</sup>&emsp;
    <a href='https://scholar.google.com.hk/citations?user=PnNAAasAAAAJ&hl=en' target='_blank'>Litong Feng</a><sup>3</sup>&emsp;
    <a href='https://www.statfe.com/' target='_blank'>Wayne Zhang</a><sup>3</sup>&emsp;
</div>
<div>
    <sup>1</sup>S-Lab, Nanyang Technological University&emsp; 
    <sup>2</sup>CCDS, Nanyang Technological University&emsp; 
    <sup>3</sup>SenseTime Research&emsp;
</div>

[![Demo](https://img.shields.io/badge/Online-Demo-red)]()
[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://mc-lan.github.io/Text4Seg/)
[![Paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](http://arxiv.org/abs/2410.09855)

</div>

<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Oryx Video-ChatGPT">
</p>


## Dependencies and Installation
Follow [ms-swift](https://github.com/modelscope/ms-swift?tab=readme-ov-file#%EF%B8%8F-installation) to prepare the environment.
```
# create new anaconda env
cd ms-swift
conda create -n swift python=3.10
conda activate swift
pip install -e .
pip install 'ms-swift[all]' -U
```

## Datasets
Hard copy or soft copy the `playground/data/refer_seg` to `ms-swift/data`.
```
cp -r playground/data/refer_seg ms-swift/data
```
Create the json file for each MLLM.
```
python data/create_refcoco_***.py
```

## Pre-trained weights
Download the pre_trained weights from the Hugging Face [deepseek-vl-1.3b-chat](https://huggingface.co/deepseek-ai/deepseek-vl-1.3b-chat), [deepseek-vl-7b-chat](https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat), [intern-vl2-8b](https://huggingface.co/OpenGVLab/InternVL2-8B), [llava-1_5_7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf), [llava-1_5_13b-hf](https://huggingface.co/llava-hf/llava-1.5-13b-hf), [Qwen-VL-7b-chat](https://huggingface.co/Qwen/Qwen-VL-Chat) to `checkpoints` folder.
For SAM, please download the checkpoints from [SAM](https://github.com/facebookresearch/segment-anything#model-checkpoints).
```
├── checkpoints
│   ├── deepseek-vl-1.3b-chat
│   ├── deepseek-vl-7b-chat
│   ├── intern-vl2-8b
│   ├── llava-1_5_7b-hf
│   ├── llava-1_5_13b-hf
│   ├── Qwen-VL-7b-chat
```

## Checkpoints

Download the checkpoints (lora weight) from [OneDrive](https://entuedu-my.sharepoint.com/:f:/g/personal/lanm0002_e_ntu_edu_sg/Eu9JQC2QDuZFj3wpNksYYfQBRnEcX10VQOWdZ3bKFr7Qkg?e=WG5vVO) to `checkpoints` folder.
```
├── checkpoints
│   ├── deepseek-vl-1.3b-chat
│   ├── deepseek-vl-7b-chat
│   ├── intern-vl2-8b
│   ├── llava-1_5_7b-hf
│   ├── llava-1_5_13b-hf
│   ├── Qwen-VL-7b-chat
|   ├──refcoco_5e_lr2e-4-bs128_r64_16
|       ├──deepseek-vl-1.3b-chat
|       ├──deepseek-vl-7b-chat
|       ├──intern-vl2-8b
|       ├──llava-1_5_7b-hf
|       ├──llava-1_5_13b-hf
|       └──Qwen-VL-7b-chat
|   └──grefcoco_2e_lr2e-4-bs128_r64_16
|       ├──deepseek-vl-7b-chat
|       ├──intern-vl2-8b
|       ├──llava-1_5_7b-hf
|       └──llava-1_5_13b-hf
```

### Quick Inference
```
python demo.py
```

### Model evaluation
Referring expression segmengtation and comprehension:
```
bash text4seg/internvl2-8B/infer.sh
```

Generalized referring expression segmengtation:
```
bash text4seg/internvl2-8B/g_infer.sh
```
Results will be saved in `output.txt`.

### Model training

Step 1: generate the json files
```
python data/create_refcoco_internvl2.py
python data/create_grefercoco_internvl2.py
```

Step 2: SFT on refercoco datasets
```
bash text4seg/internvl2-8B/train.sh
```

Step 3: SFT on grefercoco datasets
```
bash text4seg/internvl2-8B/g_train.sh
```

