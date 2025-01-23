
# CogVLM2 最佳实践

## 目录
- [环境准备](#环境准备)
- [推理](#推理)
- [微调](#微调)
- [微调后推理](#微调后推理)


## 环境准备
```shell
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'
```

模型链接:
- cogvlm2-19b-chat: [https://modelscope.cn/models/ZhipuAI/cogvlm2-llama3-chinese-chat-19B/summary](https://modelscope.cn/models/ZhipuAI/cogvlm2-llama3-chinese-chat-19B/summary)
- cogvlm2-en-19b-chat: [https://modelscope.cn/models/ZhipuAI/cogvlm2-llama3-chat-19B/summary](https://modelscope.cn/models/ZhipuAI/cogvlm2-llama3-chat-19B/summary)


## 推理

推理cogvlm2-19b-chat:
```shell
# Experimental environment: A100
# 43GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift infer --model_type cogvlm2-19b-chat
```

输出: (支持传入本地路径或URL)
```python
"""
<<< 你好
Input a media path or URL <<<
你好！我是一个人工智能助手，随时准备回答你的问题。有什么我可以帮助你的吗？
--------------------------------------------------
<<< clear
<<< 描述这张图片
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png
这是一张特写照片，展示了一只灰色和白色相间的猫。这只猫的眼睛是灰色的，鼻子是粉色的，嘴巴微微张开。它的毛发看起来柔软而蓬松，背景模糊，突出了猫的面部特征。
--------------------------------------------------
<<< clear
<<< 图中有几只羊
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png
图中有四只羊。
--------------------------------------------------
<<< clear
<<< 计算结果是多少?
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/math.png
计算结果是49556。
--------------------------------------------------
<<< clear
<<< 根据图片中的内容写首诗
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/poem.png
夜幕低垂，小船悠然，
在碧波荡漾的湖面上航行。
船头灯火，照亮前行的道路，
照亮了周围的黑暗。

湖面上的涟漪，
仿佛是无数的精灵在跳舞。
它们随着船的移动而荡漾，
为这宁静的夜晚增添了生机。

船上的乘客，
沉浸在这如诗如画的景色中。
他们欣赏着湖光山色，
感受着大自然的恩赐。

夜色渐深，小船驶向远方，
但心中的美好永远留存。
这段旅程，
让他们更加珍惜生命中的每一刻。
--------------------------------------------------
<<< clear
<<< 对图片进行OCR
Input a media path or URL <<< https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/ocr.png
图中内容是：

简介

SWIFT支持250+LLM和35+ MLLM(多模态大模型)的训练、推理、评测和部署。开发者可以直接将我们的框架应用到自己的Research和生产环境中，实现模型训练评测到应用的完整链路。我们除支持了PEFT提供的轻量训练方案外，也提供了一个完整的Adapters库以支持最新的训练技术，如NEFTune、LoRA+、LLaMA-PRO等，这个适配器库可以脱离训练脚本直接使用在自己的自定流程中。

为方便不熟悉深度学习的用户使用，我们提供了一个Gradio的web-ui用于控制训练和推理，并提供了配套的深度学习课程和最佳实践供新手入门。

此外，我们也在拓展其他模态的能力，目前我们支持了AnimateDiff的全参数训练和LoRA训练。

SWIFT具有丰富的文档体系，如有使用问题请查看这里。

可以在Huggingface space和ModelScope创空间中体验SWIFTweb-ui功能了。
"""
```

示例图片如下:

cat:

<img src="http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png" width="250" style="display: inline-block;">

animal:

<img src="http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png" width="250" style="display: inline-block;">

math:

<img src="http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/math.png" width="250" style="display: inline-block;">

poem:

<img src="http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/poem.png" width="250" style="display: inline-block;">

ocr:

<img src="https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/ocr.png" width="250" style="display: inline-block;">

**单样本推理**

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type, inference_stream
)
from swift.utils import seed_everything
import torch

model_type = ModelType.cogvlm2_19b_chat
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')

model, tokenizer = get_model_tokenizer(model_type, torch.float16,
                                       model_kwargs={'device_map': 'auto'})
model.generation_config.max_new_tokens = 256
template = get_template(template_type, tokenizer)
seed_everything(42)

images = ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png']
query = '距离各城市多远？'
response, history = inference(model, template, query, images=images)
print(f'query: {query}')
print(f'response: {response}')

# 流式
query = '距离最远的城市是哪？'
images = images
gen = inference_stream(model, template, query, history, images=images)
print_idx = 0
print(f'query: {query}\nresponse: ', end='')
for response, _ in gen:
    delta = response[print_idx:]
    print(delta, end='', flush=True)
    print_idx = len(response)
print()

"""
query: 距离各城市多远？
response: 距离马踏Mata有14km，距离阳江Yangjiang有62km，距离广州Guangzhou有293km。
query: 距离最远的城市是哪？
response: 距离最远的城市是广州Guangzhou，有293km。
"""
```

示例图片如下:

road:

<img src="http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png" width="250" style="display: inline-block;">


## 微调
多模态大模型微调通常使用**自定义数据集**进行微调. 这里展示可直接运行的demo:

```shell
# Experimental environment: A100
# 70GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type cogvlm2-19b-chat \
    --dataset coco-en-2-mini \
```

[自定义数据集](../LLM/自定义与拓展.md#-推荐命令行参数的形式)支持json, jsonl样式, 以下是自定义数据集的例子:

(支持多轮对话, 但总的轮次对话只能包含一张图片, 支持传入本地路径或URL)

```jsonl
{"query": "55555", "response": "66666", "images": ["image_path"]}
{"query": "eeeee", "response": "fffff", "history": [], "images": ["image_path"]}
{"query": "EEEEE", "response": "FFFFF", "history": [["query1", "response1"], ["query2", "response2"]], "images": ["image_path"]}
```


## 微调后推理
直接推理:
```shell
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/cogvlm2-19b-chat/vx-xxx/checkpoint-xxx \
    --load_dataset_config true \
```

**merge-lora**并推理:
```shell
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir output/cogvlm2-19b-chat/vx-xxx/checkpoint-xxx \
    --merge_lora true

CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/cogvlm2-19b-chat/vx-xxx/checkpoint-xxx-merged \
    --load_dataset_config true
```
