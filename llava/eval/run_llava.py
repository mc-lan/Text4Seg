import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

from llava.model.segment_anything import SamPredictor, sam_model_registry
from llava.eval.utils import compute_logits_from_mask, show_points, masks_sample_points


from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

import requests
from PIL import Image
from io import BytesIO
import re


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def upsample_tensor_vectorized(a, s):
    h, w = a.shape
    sh, sw = int(h * s), int(w * s)
    # Create an output tensor of zeros
    result = torch.zeros((sh, sw), dtype=a.dtype, device=a.device)
    # Calculate the target indices
    offset = int(s / 2)
    i_indices = torch.arange(h) * s + offset
    j_indices = torch.arange(w) * s + offset
    # Use broadcasting to fill the result tensor
    result[i_indices[:, None].long(), j_indices.long()] = a
    return result

def translate_sequence(sequence_str):
    """
    Translates a comma-separated sequence of categorical data to numerical labels,
    identifying categories from the sequence.

    Parameters:
    sequence_str (str): The comma-separated sequence of categorical data.

    Returns:
    list: The sequence of numerical labels.
    """
    # Split the string into a list of categories
    sequence = sequence_str.split('|')

    # strip the whitespace from each category
    sequence = [seq.strip() for seq in sequence]

    # Identify unique categories from the sequence
    unique_categories = list(dict.fromkeys(sequence))

    # place "others" at the beginning of the list
    if "others" in unique_categories:
        unique_categories.remove("others")
        unique_categories.insert(0, "others")

    # Create a dictionary to map each category to a unique integer
    category_to_label = {category: idx for idx, category in enumerate(unique_categories)}

    # Translate the sequence using the dictionary
    translated_sequence = [category_to_label[item] for item in sequence]

    return translated_sequence

def decode_mask(encoded_str):
    rows = encoded_str.strip("\n").split("\n ")
    decoded_list = []
    for row in rows:
        tokens = row.split("| ")
        for token in tokens:
            label, count = token.split(" *")
            decoded_list.extend([label] * int(count))
    return "|".join(decoded_list)

def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    sam = sam_model_registry["vit_h"](checkpoint=args.sam_path)
    sam = sam.to(dtype=torch.float32, device='cuda')
    predictor = SamPredictor(sam)

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(args)
    images = load_images(image_files)
    images_ori = images[0]
    w_ori, h_ori = images_ori.size
    images_new = []
    for image in images:
        image = image.resize((336, 336), Image.BILINEAR)
        images_new.append(image)

    image_sizes = [x.size for x in images_new]
    images = [x for x in images_new]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        predictor.set_image(np.array(images_ori))
        output_ids = model.generate(
            input_ids,
            images=[images_tensor],
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    print(outputs)

    if "<seg>" not in outputs:
        print("No mask found.")
        return

    h, w = 24, 24

    # get context between <seg> and </seg>
    mask_labels = outputs.split("<seg>")[1].split("</seg>")[0]
    mask_labels = decode_mask(mask_labels)
    pred_mask = translate_sequence(mask_labels)
    if len(pred_mask) < h * w:
        pred_mask = pred_mask + [pred_mask[-1]] * (h * w - len(pred_mask))
    elif len(pred_mask) > h * w:
        pred_mask = pred_mask[:h * w]

    mask = torch.tensor(pred_mask).reshape(h, w)

    mask_pred = F.interpolate(mask.unsqueeze(0).unsqueeze(0).double(), size=(h_ori, w_ori), mode='nearest').squeeze(0).squeeze(0)

    new_mask_pred = np.zeros((mask_pred.shape[0], mask_pred.shape[1]))
    unique_classes = np.unique(mask_pred)

    for class_id in unique_classes:
        # Skip if the class_id is the background (e.g., class 0 if it's background)
        if class_id == 0:
            continue

        # Create a binary mask for the current class
        binary_mask = (mask_pred == class_id).to(torch.float64)  # Binary mask for current class

        try:
            logits = compute_logits_from_mask(binary_mask)
            point_coords, point_labels = masks_sample_points(binary_mask)
            
            sam_mask, score, logit = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                mask_input=logits,
                multimask_output=False
            )

            for iter in range(2):
                sam_mask, score, logit = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    mask_input=logit,
                    multimask_output=False
                )
            
        except:
            # In case of an error, use a zero mask for this class
            sam_mask = np.zeros((h_ori, w_ori))

        # Add the processed mask back to the new mask for this class
        new_mask_pred[sam_mask[0] > 0] = class_id
    sam_mask = new_mask_pred

    sam_mask_s = sam_mask.astype("uint8")
    sam_mask_s = Image.fromarray(sam_mask_s).convert('L')

    sam_mask_s.save("images/horse_mask.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(np.array(images_ori))
    axes[0].set_title("Image")
    axes[0].axis('off')

    axes[1].imshow(sam_mask)
    axes[1].set_title("Mask")
    axes[1].axis('off')

    plt.tight_layout()

    # save
    plt.savefig('images/mask.png')
    print('done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/mnt/lustre/lanmengcheng.vendor/LLaVa/checkpoints/llava-v1.5-7b-lora-r128-p16-e2/")
    parser.add_argument("--model-base", type=str, default='./pre_trained/vicuna-7b-v1.5/')
    parser.add_argument("--image-file", type=str, default='./images/horses.jpg')
    parser.add_argument("--sam_path", type=str, default='llava/model/segment_anything/sam_vit_h_4b8939.pth')
    parser.add_argument("--query", type=str, default='Please segment horses in this image.')
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None) 
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=3069)
    args = parser.parse_args()

    eval_model(args)
