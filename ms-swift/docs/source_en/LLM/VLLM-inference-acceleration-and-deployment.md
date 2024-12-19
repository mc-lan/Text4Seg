# VLLM Inference Acceleration and Deployment
The models supported by vllm can be found in [Supported Models](Supported-models-datasets.md#Models).

You can check the best practices for inference acceleration and deployment of Llama 3.1 405b [here](https://github.com/modelscope/ms-swift/issues/1484).

## Table of Contents
- [Environment Preparation](#environment-preparation)
- [Inference Acceleration](#inference-acceleration)
- [Web-UI Acceleration](#web-ui-acceleration)
- [Deployment](#deployment)

## Environment Preparation
GPU devices: A10, 3090, V100, A100 are all supported.
```bash
# Set pip global mirror (speeds up downloads)
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# Install ms-swift
pip install 'ms-swift[llm]' -U

# vllm version corresponds to cuda version, please select version according to `https://docs.vllm.ai/en/latest/getting_started/installation.html`
pip install vllm
pip install openai -U

# Environment alignment (usually not needed. If you get errors, you can run the code below, the repo uses the latest environment for testing)
pip install -r requirements/framework.txt -U
pip install -r requirements/llm.txt -U
```

## Inference Acceleration

### Using Python
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    ModelType, get_vllm_engine, get_default_template_type,
    get_template, inference_vllm, inference_stream_vllm
)

model_type = ModelType.qwen_7b_chat
llm_engine = get_vllm_engine(model_type)
template_type = get_default_template_type(model_type)
template = get_template(template_type, llm_engine.hf_tokenizer)
# Similar to `transformers.GenerationConfig` interface
llm_engine.generation_config.max_new_tokens = 256
generation_info = {}

request_list = [{'query': 'Hello!'}, {'query': 'Where is the capital of Zhejiang?'}]
resp_list = inference_vllm(llm_engine, template, request_list, generation_info=generation_info)
for request, resp in zip(request_list, resp_list):
    print(f"query: {request['query']}")
    print(f"response: {resp['response']}")
print(generation_info)

# stream
history1 = resp_list[1]['history']
request_list = [{'query': 'Is there anything tasty here?', 'history': history1}]
gen = inference_stream_vllm(llm_engine, template, request_list, generation_info=generation_info)
query = request_list[0]['query']
print_idx = 0
print(f'query: {query}\nresponse: ', end='')
for resp_list in gen:
    resp = resp_list[0]
    response = resp['response']
    delta = response[print_idx:]
    print(delta, end='', flush=True)
    print_idx = len(response)
print()

history = resp_list[0]['history']
print(f'history: {history}')
print(generation_info)

"""Out[0]
query: Hello!
response: Hello! How can I assist you today? Is there something on your mind that you would like to talk about or ask me about? I'm here to help answer any questions you may have.
query: Where is the capital of Zhejiang?
response: The capital of Zhejiang is Hangzhou. It is located in eastern China, on the southern bank of the Qiantang River. Hangzhou is known for its beautiful natural scenery, historic landmarks, and cultural heritage, including the West Lake, Lingyin Temple, and the Longjing tea plantations. It is also an important economic center, with a thriving technology industry and a strong presence in finance and commerce.
{'num_prompt_tokens': 49, 'num_generated_tokens': 126, 'runtime': 1.1199991840112489, 'samples/s': 1.7857155867177066, 'tokens/s': 112.50008196321552}
query: Is there anything tasty here?
response: Yes, Hangzhou is famous for its delicious food! One of the most popular dishes from Hangzhou is the "Dongpo pork", which is made from slow-cooked pork that has been marinated in a sweet and savory sauce made from soy sauce, rice wine, and sugar. Another popular dish is "West Lake fish in vinegar sauce", which features fresh fish fillets cooked in a tangy vinegar sauce and served with steamed buns. There are many other delicious local specialties to try, such as "Longjing tea eggs" (steamed eggs boiled in Longjing tea), "Jiashan bamboo shoots" (a type of vegetable dish), and "Zhouguyu" (a soup made with fermented tofu). Hangzhou's cuisine is known for its delicate flavors and use of fresh ingredients, making it a must-try for any food lover visiting the city.
history: [['Where is the capital of Zhejiang?', 'The capital of Zhejiang is Hangzhou. It is located in eastern China, on the southern bank of the Qiantang River. Hangzhou is known for its beautiful natural scenery, historic landmarks, and cultural heritage, including the West Lake, Lingyin Temple, and the Longjing tea plantations. It is also an important economic center, with a thriving technology industry and a strong presence in finance and commerce.'], ['Is there anything tasty here?', 'Yes, Hangzhou is famous for its delicious food! One of the most popular dishes from Hangzhou is the "Dongpo pork", which is made from slow-cooked pork that has been marinated in a sweet and savory sauce made from soy sauce, rice wine, and sugar. Another popular dish is "West Lake fish in vinegar sauce", which features fresh fish fillets cooked in a tangy vinegar sauce and served with steamed buns. There are many other delicious local specialties to try, such as "Longjing tea eggs" (steamed eggs boiled in Longjing tea), "Jiashan bamboo shoots" (a type of vegetable dish), and "Zhouguyu" (a soup made with fermented tofu). Hangzhou\'s cuisine is known for its delicate flavors and use of fresh ingredients, making it a must-try for any food lover visiting the city.']]
{'num_prompt_tokens': 129, 'num_generated_tokens': 181, 'runtime': 2.307140746997902, 'samples/s': 0.43343692893518526, 'tokens/s': 78.45208413726853}
"""
```

### Using CLI
```bash
# qwen
CUDA_VISIBLE_DEVICES=0 swift infer --model_type qwen-7b-chat --infer_backend vllm
# yi
CUDA_VISIBLE_DEVICES=0 swift infer --model_type yi-6b-chat --infer_backend vllm
# gptq
CUDA_VISIBLE_DEVICES=0 swift infer --model_type qwen1half-7b-chat-int4 --infer_backend vllm

# TP
# 2 * 80GiB A100
RAY_memory_monitor_refresh_ms=0 CUDA_VISIBLE_DEVICES=0,1 swift infer \
    --model_type qwen2-72b-instruct --infer_backend vllm --max_model_len 2048 \
    --tensor_parallel_size 2
```

### Fine-tuned Models

**Single sample inference**:

For models fine-tuned using LoRA, you need to first [merge-lora](LLM-fine-tuning.md#merge-lora) to generate a complete checkpoint directory.

Models fine-tuned with full parameters can seamlessly use VLLM for inference acceleration.
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    ModelType, get_vllm_engine, get_default_template_type,
    get_template, inference_vllm
)

ckpt_dir = 'vx-xxx/checkpoint-100-merged'
model_type = ModelType.qwen_7b_chat
template_type = get_default_template_type(model_type)

llm_engine = get_vllm_engine(model_type, model_id_or_path=ckpt_dir)
tokenizer = llm_engine.hf_tokenizer
template = get_template(template_type, tokenizer)
query = 'Hello'
resp = inference_vllm(llm_engine, template, [{'query': query}])[0]
print(f"response: {resp['response']}")
print(f"history: {resp['history']}")
```

**Using CLI**:
```bash
# merge LoRA incremental weights and use vllm for inference acceleration
# If you need quantization, you can specify `--quant_bits 4`.
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx' --merge_lora true

# Evaluate using dataset
# If you want to infer all dataset samples, please additionally specify `--show_dataset_sample -1`.
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx-merged' \
    --infer_backend vllm \
    --load_dataset_config true \

# Manual evaluation
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx-merged' \
    --infer_backend vllm \
```

## Web-UI Acceleration

### Original Models
```bash
CUDA_VISIBLE_DEVICES=0 swift app-ui --model_type qwen-7b-chat --infer_backend vllm
```

### Fine-tuned Models
```bash
# merge LoRA incremental weights and use vllm as backend to build app-ui
# If you need quantization, you can specify `--quant_bits 4`.
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx' --merge_lora true

CUDA_VISIBLE_DEVICES=0 swift app-ui --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx-merged' --infer_backend vllm
```

## Deployment
Swift uses VLLM as the inference backend and is compatible with the OpenAI API style.

For server deployment command line arguments, refer to: [deploy command line arguments](Command-line-parameters.md#deploy-Parameters).

For OpenAI API arguments on the client side, refer to: https://platform.openai.com/docs/api-reference/introduction.

### Original Models
#### qwen-7b-chat

**Server side:**
```bash
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type qwen-7b-chat
# Multi-GPU deployment
RAY_memory_monitor_refresh_ms=0 CUDA_VISIBLE_DEVICES=0,1,2,3 swift deploy --model_type qwen-7b-chat --tensor_parallel_size 4
```

**Client side:**

Test:
```bash
curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "qwen-7b-chat",
"messages": [{"role": "user", "content": "What to do if I can't fall asleep at night?"}],
"max_tokens": 256,
"temperature": 0
}'
```

Synchronous client interface using swift:
```python
from swift.llm import get_model_list_client, XRequestConfig, inference_client

model_list = get_model_list_client()
model_type = model_list.data[0].id
print(f'model_type: {model_type}')

query = 'Where is the capital of Zhejiang?'
request_config = XRequestConfig(seed=42)
resp = inference_client(model_type, query, request_config=request_config)
response = resp.choices[0].message.content
print(f'query: {query}')
print(f'response: {response}')

history = [(query, response)]
query = 'What delicious food is there?'
request_config = XRequestConfig(stream=True, seed=42)
stream_resp = inference_client(model_type, query, history, request_config=request_config)
print(f'query: {query}')
print('response: ', end='')
for chunk in stream_resp:
    print(chunk.choices[0].delta.content, end='', flush=True)
print()

"""Out[0]
model_type: qwen-7b-chat
query: Where is the capital of Zhejiang?
response: The capital of Zhejiang Province is Hangzhou.
query: What delicious food is there?
response: Hangzhou has many delicious foods, such as West Lake Vinegar Fish, Dongpo Pork, Longjing Shrimp, Beggar's Chicken, etc. In addition, Hangzhou also has many specialty snacks, such as West Lake Lotus Root Powder, Hangzhou Xiao Long Bao, Hangzhou You Tiao, etc.
"""
```

Asynchronous client interface using swift:
```python
import asyncio
from swift.llm import get_model_list_client, XRequestConfig, inference_client_async

model_list = get_model_list_client()
model_type = model_list.data[0].id
print(f'model_type: {model_type}')

query = 'Where is the capital of Zhejiang?'
request_config = XRequestConfig(seed=42)
tasks = [inference_client_async(model_type, query, request_config=request_config) for _ in range(5)]
async def _batch_run(tasks):
    return await asyncio.gather(*tasks)

resp_list = asyncio.run(_batch_run(tasks))
resp = resp_list[0]
response = resp.choices[0].message.content
print(f'query: {query}')
print(f'response: {response}')

async def _stream():
    global query
    history = [(query, response)]
    query = 'What delicious food is there?'
    request_config = XRequestConfig(stream=True, seed=42)
    stream_resp = await inference_client_async(model_type, query, history, request_config=request_config)
    print(f'query: {query}')
    print('response: ', end='')
    async for chunk in stream_resp:
        print(chunk.choices[0].delta.content, end='', flush=True)
    print()

asyncio.run(_stream())
"""Out[0]
model_type: qwen-7b-chat
query: Where is the capital of Zhejiang?
response: The capital of Zhejiang is Hangzhou.
query: What delicious food is there?
response: Hangzhou is famous for its delicious food, such as West Lake Fish in Vinegar Gravy, Dongpo Pork, and Longjing Tea.
"""
```

Using OpenAI (synchronous):
```python
from openai import OpenAI
client = OpenAI(
    api_key='EMPTY',
    base_url='http://localhost:8000/v1',
)
model_type = client.models.list().data[0].id
print(f'model_type: {model_type}')

query = 'Where is the capital of Zhejiang?'
messages = [{
    'role': 'user',
    'content': query
}]
resp = client.chat.completions.create(
    model=model_type,
    messages=messages,
    seed=42)
response = resp.choices[0].message.content
print(f'query: {query}')
print(f'response: {response}')

# Streaming
messages.append({'role': 'assistant', 'content': response})
query = 'What delicious food is there?'
messages.append({'role': 'user', 'content': query})
stream_resp = client.chat.completions.create(
    model=model_type,
    messages=messages,
    stream=True,
    seed=42)

print(f'query: {query}')
print('response: ', end='')
for chunk in stream_resp:
    print(chunk.choices[0].delta.content, end='', flush=True)
print()

"""Out[0]
model_type: qwen-7b-chat
query: Where is the capital of Zhejiang?
response: The capital of Zhejiang Province is Hangzhou.
query: What delicious food is there?
response: Hangzhou has many delicious foods, such as West Lake Vinegar Fish, Dongpo Pork, Longjing Shrimp, Beggar's Chicken, etc. In addition, Hangzhou also has many specialty snacks, such as West Lake Lotus Root Powder, Hangzhou Xiao Long Bao, Hangzhou You Tiao, etc.
"""
```

#### qwen-7b

**Server side:**
```bash
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type qwen-7b
# Multi-GPU deployment
RAY_memory_monitor_refresh_ms=0 CUDA_VISIBLE_DEVICES=0,1,2,3 swift deploy --model_type qwen-7b --tensor_parallel_size 4
```

**Client side:**

Test:
```bash
curl http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "qwen-7b",
"prompt": "Zhejiang -> Hangzhou\nAnhui -> Hefei\nSichuan ->",
"max_tokens": 32,
"temperature": 0.1,
"seed": 42
}'
```

Synchronous client interface using swift:
```python
from swift.llm import get_model_list_client, XRequestConfig, inference_client

model_list = get_model_list_client()
model_type = model_list.data[0].id
print(f'model_type: {model_type}')

query = 'Zhejiang -> Hangzhou\nAnhui -> Hefei\nSichuan ->'
request_config = XRequestConfig(max_tokens=32, temperature=0.1, seed=42)
resp = inference_client(model_type, query, request_config=request_config)
response = resp.choices[0].text
print(f'query: {query}')
print(f'response: {response}')

request_config.stream = True
stream_resp = inference_client(model_type, query, request_config=request_config)
print(f'query: {query}')
print('response: ', end='')
for chunk in stream_resp:
    print(chunk.choices[0].text, end='', flush=True)
print()

"""Out[0]
model_type: qwen-7b
query: Zhejiang -> Hangzhou
Anhui -> Hefei
Sichuan ->
response:  Chengdu
Guangdong -> Guangzhou
Jiangsu -> Nanjing
Zhejiang -> Hangzhou
Anhui -> Hefei
Sichuan -> Chengdu

query: Zhejiang -> Hangzhou
Anhui -> Hefei
Sichuan ->
response:  Chengdu
Guangdong -> Guangzhou
Jiangsu -> Nanjing
Zhejiang -> Hangzhou
Anhui -> Hefei
Sichuan -> Chengdu
"""
```

Asynchronous client interface using swift:
```python
import asyncio
from swift.llm import get_model_list_client, XRequestConfig, inference_client_async

model_list = get_model_list_client()
model_type = model_list.data[0].id
print(f'model_type: {model_type}')

query = 'Zhejiang -> Hangzhou\nAnhui -> Hefei\nSichuan ->'
request_config = XRequestConfig(max_tokens=32, temperature=0.1, seed=42)

resp = asyncio.run(inference_client_async(model_type, query, request_config=request_config))
response = resp.choices[0].text
print(f'query: {query}')
print(f'response: {response}')

async def _stream():
    request_config.stream = True
    stream_resp = await inference_client_async(model_type, query, request_config=request_config)
    print(f'query: {query}')
    print('response: ', end='')
    async for chunk in stream_resp:
        print(chunk.choices[0].text, end='', flush=True)
    print()

asyncio.run(_stream())
"""Out[0]
model_type: qwen-7b
query: Zhejiang -> Hangzhou
Anhui -> Hefei
Sichuan ->
response:  Chengdu
Guangdong -> Guangzhou
Jiangsu -> Nanjing
Zhejiang -> Hangzhou
Anhui -> Hefei
Sichuan -> Chengdu

query: Zhejiang -> Hangzhou
Anhui -> Hefei
Sichuan ->
response:  Chengdu
Guangdong -> Guangzhou
Jiangsu -> Nanjing
Zhejiang -> Hangzhou
Anhui -> Hefei
Sichuan -> Chengdu
"""
```

Using OpenAI (synchronous):
```python
from openai import OpenAI
client = OpenAI(
    api_key='EMPTY',
    base_url='http://localhost:8000/v1',
)
model_type = client.models.list().data[0].id
print(f'model_type: {model_type}')

query = 'Zhejiang -> Hangzhou\nAnhui -> Hefei\nSichuan ->'
kwargs = {'model': model_type, 'prompt': query, 'seed': 42, 'temperature': 0.1, 'max_tokens': 32}

resp = client.completions.create(**kwargs)
response = resp.choices[0].text
print(f'query: {query}')
print(f'response: {response}')

# Streaming
stream_resp = client.completions.create(stream=True, **kwargs)
response = resp.choices[0].text
print(f'query: {query}')
print('response: ', end='')
for chunk in stream_resp:
    print(chunk.choices[0].text, end='', flush=True)
print()

"""Out[0]
model_type: qwen-7b
query: Zhejiang -> Hangzhou
Anhui -> Hefei
Sichuan ->
response:  Chengdu
Guangdong -> Guangzhou
Jiangsu -> Nanjing
Zhejiang -> Hangzhou
Anhui -> Hefei
Sichuan -> Chengdu

query: Zhejiang -> Hangzhou
Anhui -> Hefei
Sichuan ->
response:  Chengdu
Guangdong -> Guangzhou
Jiangsu -> Nanjing
Zhejiang -> Hangzhou
Anhui -> Hefei
Sichuan -> Chengdu
"""
```

### Fine-tuned Models
Server side:
```bash
# merge LoRA incremental weights and deploy
# If you need quantization, you can specify `--quant_bits 4`.
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx' --merge_lora true

CUDA_VISIBLE_DEVICES=0 swift deploy --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx-merged'
```

The example code for the client side is the same as the original models.

## Multiple LoRA Deployments

The current model deployment method now supports multiple LoRA deployments with `peft>=0.10.0`. The specific steps are:

- Ensure `merge_lora` is set to `False` during deployment.
- Use the `--lora_modules` argument, which can be referenced in the [command line documentation](Command-line-parameters.md).
- Specify the name of the LoRA tuner in the model field during inference.

Example:

```shell
# Assuming a LoRA model named Kakarot was trained from llama3-8b-instruct
# Server side
swift deploy --ckpt_dir /mnt/ckpt-1000 --infer_backend pt --lora_modules my_tuner=/mnt/my-tuner
# This loads two tuners, one is `default-lora` from `/mnt/ckpt-1000`, and the other is `my_tuner` from `/mnt/my-tuner`

# Client side
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
"model": "my-tuner",
"messages": [{"role": "user", "content": "who are you?"}],
"max_tokens": 256,
"temperature": 0
}'
# resp: I am Kakarot...
# If the mode='llama3-8b-instruct' is specified, it will return I'm llama3..., which is the response of the original model
```

> [!NOTE]
>
> If the `--ckpt_dir` parameter is a LoRA path, the original default will be loaded onto the default-lora tuner, and other tuners need to be loaded through `lora_modules` manually.

## VLLM & LoRA

Models supported by VLLM & LoRA can be viewed at: https://docs.vllm.ai/en/latest/models/supported_models.html

### Setting Up LoRA

```shell
# Experimental environment: 4 * A100
# 4 * 30GB GPU memory
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
swift sft \
    --model_type llama2-7b-chat \
    --dataset self-cognition#500 sharegpt-gpt4:default#1000 \
    --max_length 4096 \
    --learning_rate 1e-4 \
    --output_dir output \
    --lora_target_modules ALL \
    --model_name 'Xiao Huang' \
    --model_author ModelScope \
```


### Accelerating VLLM Inference

Inference:

```shell
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/llama2-7b-chat/vx-xxx/checkpoint-xxx \
    --infer_backend vllm \
    --vllm_enable_lora true
```

Inference results:

```python
"""
<<< who are you?
I am an artificial intelligence language model developed by ModelScope. I am designed to assist and communicate with users in a helpful and respectful manner. I can answer questions, provide information, and engage in conversation. How can I help you?
"""
```

Single sample inference:

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from swift.llm import (
    ModelType, get_vllm_engine, get_default_template_type,
    get_template, inference_stream_vllm, LoRARequest, inference_vllm
)

lora_checkpoint = 'output/llama2-7b-chat/vx-xxx/checkpoint-xxx'
lora_request = LoRARequest('default-lora', 1, lora_checkpoint)

model_type = ModelType.llama2_7b_chat
llm_engine = get_vllm_engine(model_type, torch.float16, enable_lora=True,
                             max_loras=1, max_lora_rank=16)
template_type = get_default_template_type(model_type)
template = get_template(template_type, llm_engine.hf_tokenizer)
# Interface similar to `transformers.GenerationConfig`
llm_engine.generation_config.max_new_tokens = 256

# using lora
request_list = [{'query': 'who are you?'}]
query = request_list[0]['query']
resp_list = inference_vllm(llm_engine, template, request_list, lora_request=lora_request)
response = resp_list[0]['response']
print(f'query: {query}')
print(f'response: {response}')

# without lora
gen = inference_stream_vllm(llm_engine, template, request_list)
query = request_list[0]['query']
print(f'query: {query}\nresponse: ', end='')
print_idx = 0
for resp_list in gen:
    response = resp_list[0]['response']
    print(response[print_idx:], end='', flush=True)
    print_idx = len(response)
print()
"""
query: who are you?
response: I am an artificial intelligence language model developed by ModelScope. I can understand and respond to text-based questions and prompts, and provide information and assistance on a wide range of topics.
query: who are you?
response:  Hello! I'm just an AI assistant, here to help you with any questions or tasks you may have. I'm designed to be helpful, respectful, and honest in my responses, and I strive to provide socially unbiased and positive answers. I'm not a human, but a machine learning model trained on a large dataset of text to generate responses to a wide range of questions and prompts. I'm here to help you in any way I can, while always ensuring that my answers are safe and respectful. Is there anything specific you'd like to know or discuss?
"""
```


### Deployment

**Server**:

```shell
CUDA_VISIBLE_DEVICES=0 swift deploy \
    --ckpt_dir output/llama2-7b-chat/vx-xxx/checkpoint-xxx \
    --infer_backend vllm \
    --vllm_enable_lora true
```

**Client**:

```bash
curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "default-lora",
"messages": [{"role": "user", "content": "who are you?"}],
"max_tokens": 256,
"temperature": 0
}'

curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "llama2-7b-chat",
"messages": [{"role": "user", "content": "who are you?"}],
"max_tokens": 256,
"temperature": 0
}'
```

Output:

```python
"""
{"model":"default-lora","choices":[{"index":0,"message":{"role":"assistant","content":"I am an artificial intelligence language model developed by ModelScope. I am designed to assist and communicate with users in a helpful, respectful, and honest manner. I can answer questions, provide information, and engage in conversation. How can I assist you?"},"finish_reason":"stop"}],"usage":{"prompt_tokens":141,"completion_tokens":53,"total_tokens":194},"id":"chatcmpl-fb95932dcdab4ce68f4be49c9946b306","object":"chat.completion","created":1710820459}

{"model":"llama2-7b-chat","choices":[{"index":0,"message":{"role":"assistant","content":" Hello! I'm just an AI assistant, here to help you with any questions or concerns you may have. I'm designed to provide helpful, respectful, and honest responses, while ensuring that my answers are socially unbiased and positive in nature. I'm not capable of providing harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, and I will always do my best to explain why I cannot answer a question if it does not make sense or is not factually coherent. If I don't know the answer to a question, I will not provide false information. My goal is to assist and provide accurate information to the best of my abilities. Is there anything else I can help you with?"},"finish_reason":"stop"}],"usage":{"prompt_tokens":141,"completion_tokens":163,"total_tokens":304},"id":"chatcmpl-d867a3a52bb7451588d4f73e1df4ba95","object":"chat.completion","created":1710820557}
"""
```

With openai:

```python
from openai import OpenAI
client = OpenAI(
    api_key='EMPTY',
    base_url='http://localhost:8000/v1',
)
model_type_list = [model.id for model in client.models.list().data]
print(f'model_type_list: {model_type_list}')

query = 'who are you?'
messages = [{
    'role': 'user',
    'content': query
}]
resp = client.chat.completions.create(
    model='default-lora',
    messages=messages,
    seed=42)
response = resp.choices[0].message.content
print(f'query: {query}')
print(f'response: {response}')

# stream
stream_resp = client.chat.completions.create(
    model='llama2-7b-chat',
    messages=messages,
    stream=True,
    seed=42)

print(f'query: {query}')
print('response: ', end='')
for chunk in stream_resp:
    print(chunk.choices[0].delta.content, end='', flush=True)
print()

"""Out[0]
model_type_list: ['llama2-7b-chat', 'default-lora']
query: who are you?
response: I am an artificial intelligence language model developed by ModelScope. I am designed to assist and communicate with users in a helpful, respectful, and honest manner. I can answer questions, provide information, and engage in conversation. How can I assist you?
query: who are you?
response:  Hello! I'm just an AI assistant, here to help you with any questions or concerns you may have. I'm designed to provide helpful, respectful, and honest responses, while ensuring that my answers are socially unbiased and positive in nature. I'm not capable of providing harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, and I will always do my best to explain why I cannot answer a question if it does not make sense or is not factually coherent. If I don't know the answer to a question, I will not provide false information. Is there anything else I can help you with?
"""
```
