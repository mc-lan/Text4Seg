import argparse
import torch
import os
from tqdm import tqdm
import random
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

from llava.model.segment_anything import SamPredictor, sam_model_registry
from llava.eval.utils import compute_logits_from_mask, masks_sample_points, decode_mask, translate_sequence

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.eval.refer_seg_dataset import ValDataset
from llava.eval.question_answer_list import QUESTION_PARTIAL, QUESTION_CONDITION

from torch.utils.data import Dataset, DataLoader
from llava.eval.semantic_segmentation_dataset import ADE20KDataset, PAS20Dataset, PC59Dataset, PC459Dataset

import math
import sys

ROWS = 1
COLS = 1  # COLS > ROWS
SUB_IMAGE_SIZE = 336  # Size of sub-images


def get_chunk(ds, n, k):
    chunk_size = math.ceil(len(ds) / n)  # integer division
    i = chunk_size * k
    ds.file_list = ds.file_list[i:i + chunk_size]
    return ds

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, sub_dataset, tokenizer, image_processor, model_config, dataset_split):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.dataset = sub_dataset
        self.dataset_split = dataset_split
        if "ADE20K" in args.dataset_split:
            labels = []
            with open("/mnt/lustre/lanmengcheng.vendor/LLaVa/playground/data/create_json/cls_ade20k_ori.txt") as f:
                for idx, line in enumerate(f):
                    labels.append(line.strip())
            self.labels = labels
            self.question = '|'.join(labels)
        elif "PAS20" in args.dataset_split:
            labels = []
            with open("/mnt/lustre/lanmengcheng.vendor/LLaVa/playground/data/create_json/cls_pas20.txt") as f:
                for idx, line in enumerate(f):
                    labels.append(line.strip())
            self.labels = labels
            self.question = '|'.join(labels)
        elif "PC59" in args.dataset_split:
            labels = []
            with open("/mnt/lustre/lanmengcheng.vendor/LLaVa/playground/data/create_json/cls_pc59.txt") as f:
                for idx, line in enumerate(f):
                    labels.append(line.strip())
            self.labels = labels
            self.question = '|'.join(labels)


    def __getitem__(self, index):
        image, mask, image_name = self.dataset[index]

        qs = random.choice(QUESTION_CONDITION).replace("[class_name]", self.question)

        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        w, h = image.size
        new_w, new_h = (SUB_IMAGE_SIZE * COLS, SUB_IMAGE_SIZE * ROWS) if w > h else (SUB_IMAGE_SIZE * ROWS, SUB_IMAGE_SIZE * COLS)
        image_new = image.resize((new_w, new_h), Image.BILINEAR)

        sub_images = []
        rows, cols = (COLS, ROWS) if w <= h else (ROWS, COLS)

        for i in range(rows):
            for j in range(cols):
                left = j * SUB_IMAGE_SIZE
                upper = i * SUB_IMAGE_SIZE
                crop = image_new.crop((left, upper, left + SUB_IMAGE_SIZE, upper + SUB_IMAGE_SIZE))
                sub_images.append(crop)

        image_tensor = process_images(sub_images, self.image_processor, self.model_config)
        input_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        images_size = (new_h, new_w)

        return input_id, image_tensor, images_size, image, mask, image_name

    def __len__(self):
        return len(self.dataset)


def collate_fn(batch):
    input_id, image_tensor, images_size, image, mask, image_name = zip(*batch)
    return input_id, image_tensor, images_size, image, mask, image_name


# DataLoader
def create_data_loader(sub_dataset, tokenizer, image_processor, model_config, dataset_split, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(sub_dataset, tokenizer, image_processor, model_config, dataset_split)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader, dataset.labels


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    sam = sam_model_registry["vit_h"](checkpoint=args.sam_path)
    sam = sam.to(dtype=torch.float32, device='cuda')
    predictor = SamPredictor(sam)

    if "ADE20K" in args.dataset_split:
        images_path = "/mnt/lustre/lanmengcheng.vendor/LLaVa/playground/data/semantic_seg/ADE20K/images"
        annotations_path = "/mnt/lustre/lanmengcheng.vendor/LLaVa/playground/data/semantic_seg/ADE20K/annotations"
        val_dataset = ADE20KDataset(images_path, annotations_path, file_list_path=None)
    elif "PAS20" in args.dataset_split:
        images_path = "/mnt/lustre/lanmengcheng.vendor/LLaVa/playground/data/semantic_seg/PAS20/JPEGImages"
        annotations_path = "/mnt/lustre/lanmengcheng.vendor/LLaVa/playground/data/semantic_seg/PAS20/SegmentationClass"
        file_list_path = "/mnt/lustre/lanmengcheng.vendor/LLaVa/playground/data/semantic_seg/PAS20/val.txt"
        val_dataset =PAS20Dataset(images_path, annotations_path, file_list_path=file_list_path)
    elif "PC59" in args.dataset_split:
        images_path = "/mnt/lustre/lanmengcheng.vendor/LLaVa/playground/data/semantic_seg/PC59/JPEGImages"
        annotations_path = "/mnt/lustre/lanmengcheng.vendor/LLaVa/playground/data/semantic_seg/PC59/SegmentationClassContext"
        file_list_path = "/mnt/lustre/lanmengcheng.vendor/LLaVa/playground/data/semantic_seg/PC59/pascalcontext_val.txt"
        val_dataset =PC59Dataset(images_path, annotations_path, file_list_path=file_list_path)

    sub_dataset = get_chunk(val_dataset, args.num_chunks, args.chunk_idx)

    data_loader, labels_list = create_data_loader(sub_dataset, tokenizer, image_processor, model.config, args.dataset_split)
    labels_set = {category: idx for idx, category in enumerate(labels_list)}

    if "p16" in args.model_path:
        h, w = 16, 16
    if "p24" in args.model_path:
        h, w = 24, 24
    
    jj = 0
    for (input_id, image_tensor, images_size, image, mask, image_name) in tqdm(data_loader, total=len(data_loader)):
        jj +=1
        input_id = input_id[0]
        image_tensor = image_tensor[0]
        images_size = images_size[0]
        mask = mask[0]
        image = image[0]
        image_name = image_name[0]

        mask_pred = torch.zeros((images_size[0], images_size[1]), dtype=torch.long)
        masks_pred = []

        with torch.inference_mode():
            w_ori, h_ori = image.size
            predictor.set_image(np.array(image))
            for i in range(image_tensor.size(0)):
                output_ids = model.generate(
                    input_id.unsqueeze(0).to(device='cuda', non_blocking=True),
                    images=[image_tensor[i].to(dtype=torch.float16, device='cuda', non_blocking=True)],
                    image_sizes=images_size,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True)

                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

                # get context between <seg> and </seg>
                try:
                    mask_labels = outputs.split("<seg>")[1].split("</seg>")[0]
                    mask_labels = decode_mask(mask_labels)
                    pred_mask = translate_sequence(mask_labels, labels_set)
                except:
                    print('unsuccessful!', flush=True)
                    print(outputs, flush=True)
                    pred_mask = [0] * h * w

                # if the length of pred_mask is smaller than h * w, fill the rest with the last label
                if len(pred_mask) < h * w:
                    pred_mask = pred_mask + [pred_mask[-1]] * (h * w - len(pred_mask))
                elif len(pred_mask) > h * w:
                    pred_mask = pred_mask[:h * w]

                pred_mask = torch.tensor(pred_mask).reshape(h, w)

                mask_upsample = F.interpolate(pred_mask.unsqueeze(0).unsqueeze(0).double(), size=(336, 336),mode='nearest').squeeze(0).squeeze(0)

                masks_pred.append(mask_upsample)

            rows, cols = (COLS, ROWS) if h_ori >= w_ori else (ROWS, COLS)
            for i in range(rows):
                for j in range(cols):
                    left = j * SUB_IMAGE_SIZE
                    upper = i * SUB_IMAGE_SIZE
                    mask_pred[upper:upper + SUB_IMAGE_SIZE, left:left + SUB_IMAGE_SIZE] = masks_pred[i * cols + j]
            
            mask_pred = F.interpolate(mask_pred.unsqueeze(0).unsqueeze(0).double(), size=(h_ori, w_ori),mode='nearest').squeeze(0).squeeze(0)
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

            pred_mask = mask_pred.cpu().numpy().astype("uint8")
            pred_mask = Image.fromarray(pred_mask).convert('L')
            sam_mask = sam_mask.astype("uint8")
            sam_mask = Image.fromarray(sam_mask).convert('L')

            gt_mask = mask

            image_path = os.path.join(args.save_file, model_name, args.dataset_split, image_name)
            if not os.path.exists(image_path):
                os.makedirs(image_path)
            pred_mask.save(os.path.join(image_path, f"pred_mask.png"))
            gt_mask.save(os.path.join(image_path, f"gt_mask.png"))
            sam_mask.save(os.path.join(image_path, f"sam_mask.png"))
            image.save(os.path.join(image_path, f"image.png"))

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--sam_path", type=str, default="./llava/model/segment_anything/sam_vit_h_4b8939.pth")
    parser.add_argument("--dataset_split", type=str, default="PAS20")
    parser.add_argument("--save_file", type=str, default="llava/eval/semantic_seg_results/")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=3069)
    args = parser.parse_args()

    eval_model(args)
