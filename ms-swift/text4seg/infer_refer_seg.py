import argparse
import torch
import os
from tqdm import tqdm
import random
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

from text4seg.segment_anything import SamPredictor, sam_model_registry
from text4seg.utils import compute_logits_from_mask, masks_sample_points, decode_mask, translate_sequence

from data.refer_seg_dataset import ValDataset
from data.grefer_seg_dataset import grefcocoValDataset
from data.question_answer_list import QUESTION_PARTIAL

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type, inference_stream
)
from swift.utils import seed_everything

from torch.utils.data import Dataset, DataLoader

import math
import time
import sys

def get_chunk(ds, n, k):
    chunk_size = math.ceil(len(ds) / n)  # integer division
    i = chunk_size * k
    ds.refer_seg_ds["images"] = ds.refer_seg_ds["images"][i:i + chunk_size]
    return ds

def gget_chunk(ds, n, k):
    chunk_size = math.ceil(len(ds) / n)  # integer division
    i = chunk_size * k
    ds.loaded_images = ds.loaded_images[i:i + chunk_size]
    return ds


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, args, sub_dataset):
        self.dataset = sub_dataset

    def __getitem__(self, index):
        image, masks, questions, image_path = self.dataset[index]
        image_name = os.path.basename(image_path).split(".")[0]
        if "qwen" in args.model_type:
            questions= ["<img>" + image_path + "</img>" + random.choice(QUESTION_PARTIAL).replace("[class_name]", question) for question in questions]
        else:
            questions= [random.choice(QUESTION_PARTIAL).replace("[class_name]", question) for question in questions]
        
        return image, masks, image_name, questions, image_path

    def __len__(self):
        return len(self.dataset)


def collate_fn(batch):
    image, masks, image_name, questions, image_path = zip(*batch)
    return image, masks, image_name, questions, image_path


# DataLoader
def create_data_loader(args, sub_dataset, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(args, sub_dataset)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                             collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
    model_type = args.model_type
    template_type = get_default_template_type(model_type)
    print(f'template_type: {template_type}')

    model, tokenizer = get_model_tokenizer(model_type, torch.float16,
                                           model_id_or_path=args.model_id_or_path,
                                           model_kwargs={'device_map': 'auto'})
    model.generation_config.max_new_tokens = 3069
    model.generation_config.do_sample = False
    model.generation_config.temperature = 0.0
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    template = get_template(template_type, tokenizer)
    seed_everything(42)

    sam = sam_model_registry["vit_h"](checkpoint=args.sam_path)
    sam = sam.to(dtype=torch.float32, device='cuda')
    predictor = SamPredictor(sam)
    if "grefcoco" in args.dataset_split:
        val_dataset = grefcocoValDataset(args.image_folder, args.dataset_split)
        sub_dataset = gget_chunk(val_dataset, args.num_chunks, args.chunk_idx)
    else:
        val_dataset = ValDataset(args.image_folder, args.dataset_split)
        sub_dataset = get_chunk(val_dataset, args.num_chunks, args.chunk_idx)

    data_loader = create_data_loader(args, sub_dataset)

    h = w = args.visual_tokens

    jj = 0
    for (image, masks, image_name, questions, image_path) in tqdm(data_loader, total=len(data_loader)):

        masks = masks[0]
        image = image[0]
        image_name = image_name[0]
        questions = questions[0]

        with torch.inference_mode():
            w_ori, h_ori = image.size
            predictor.set_image(np.array(image))
            for i in range(len(questions)):
                jj = jj + 1

                if "qwen" in args.model_type:
                    response, _ = inference(model, template, questions[i])
                else:
                    response, _ = inference(model, template, questions[i], images=image)
                outputs = response.strip()


                # get context between <seg> and </seg>
                try:
                    mask_labels = outputs.split("<seg>")[1].split("</seg>")[0]
                    mask_labels = decode_mask(mask_labels)
                    pred_mask = translate_sequence(mask_labels)
                    # continue
                except:
                    print(questions[i])
                    print("\n")
                    print(outputs)
                    print(jj, ': unsuccessfully!')
                    pred_mask = [0] * h * w

                # if the length of pred_mask is smaller than h * w, fill the rest with the last label
                if len(pred_mask) < h * w:
                    pred_mask = pred_mask + [pred_mask[-1]] * (h * w - len(pred_mask))
                elif len(pred_mask) > h * w:
                    pred_mask = pred_mask[:h * w]

                pred_mask = torch.tensor(pred_mask).reshape(h, w)

                # ensuare the pred_mask is 0 or 1
                pred_mask = torch.where(pred_mask > 0, torch.tensor(1), torch.tensor(0))

                mask_upsample = F.interpolate(pred_mask.unsqueeze(0).unsqueeze(0).double(), size=(h_ori, w_ori),
                                              mode='nearest').squeeze(0).squeeze(0)

                # check if 1 in the pred_mask
                if 1 not in pred_mask:
                    sam_mask = np.zeros((1, h_ori, w_ori))
                else:
                    logits = compute_logits_from_mask(mask_upsample)
                    point_coords, point_labels = masks_sample_points(mask_upsample)
                    sam_mask, score, logit = predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        mask_input=logits,
                        multimask_output=False,
                    )
                    for iter in range(2):
                        sam_mask, score, logit = predictor.predict(
                            point_coords=point_coords,
                            point_labels=point_labels,
                            mask_input=logit,
                            multimask_output=False,
                        )

                gt_mask = masks[i]
                sam_mask = sam_mask[0]

                # # save the pred_mask, sam_mask and gt_mask
                pred_mask_s = mask_upsample.cpu().numpy().astype("uint8") * 255
                sam_mask_s = sam_mask.astype("uint8") * 255
                gt_mask_s = gt_mask.cpu().numpy().astype("uint8") * 255
                pred_mask_s = Image.fromarray(pred_mask_s).convert('L')
                sam_mask_s = Image.fromarray(sam_mask_s).convert('L')
                gt_mask_s = Image.fromarray(gt_mask_s).convert('L')

                # Initialize ImageDraw
                # draw_pred = ImageDraw.Draw(pred_mask_s)
                # draw_sam = ImageDraw.Draw(sam_mask_s)
                # draw_gt = ImageDraw.Draw(gt_mask_s)

                # Add text to image using a default font
                # draw_pred.text((10, 10), questions[i], fill="white")
                # draw_sam.text((10, 10), questions[i], fill="white")
                # draw_gt.text((10, 10), questions[i], fill="white")

                ds_split = args.dataset_split.replace("|", "_")

                image_path = os.path.join(args.save_file, model_type, ds_split, image_name)
                if not os.path.exists(image_path):
                    os.makedirs(image_path)
                pred_mask_s.save(os.path.join(image_path, f"{i}_pred_mask.png"))
                sam_mask_s.save(os.path.join(image_path, f"{i}_sam_mask.png"))
                gt_mask_s.save(os.path.join(image_path, f"{i}_gt_mask.png"))
                image.save(os.path.join(image_path, f"{i}_image.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="internvl2-8b")
    parser.add_argument("--model_id_or_path", type=str, default="./output_new/list_refcoco_clef_5e_lr2e-4_bs128_r64_224/internvl2-8b/v0-20240910-021121/checkpoint-33500-merged")
    parser.add_argument("--image-folder", type=str, default="/mnt/lustre/lanmengcheng.vendor/LLaVa/playground/data/refer_seg/")
    parser.add_argument("--sam_path", type=str, default="/mnt/lustre/lanmengcheng.vendor/LLaVa/llava/model/segment_anything/sam_vit_h_4b8939.pth")
    parser.add_argument("--dataset_split", type=str, default="refcocog|umd|test")
    parser.add_argument("--save_file", type=str, default="output_eval/checkpoints_5e_lr_2e-4_bs128_r64/")
    parser.add_argument("--visual_tokens", type=int, default=24)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()

    eval_model(args)
