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
from llava.eval.utils import compute_logits_from_mask, masks_sample_points, translate_sequence, decode_mask

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.eval.refer_seg_dataset import ValDataset
from llava.eval.question_answer_list import QUESTION_PARTIAL

from torch.utils.data import Dataset, DataLoader

import math


def get_chunk(ds, n, k):
    chunk_size = math.ceil(len(ds) / n)  # integer division
    i = chunk_size * k
    ds.refer_seg_ds["images"] = ds.refer_seg_ds["images"][i:i + chunk_size]
    return ds

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, sub_dataset, tokenizer, image_processor, model_config):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.dataset = sub_dataset

    def __getitem__(self, index):
        image, masks, questions, image_path = self.dataset[index]
        image_name = os.path.basename(image_path).split(".")[0]
        input_ids, images_tensor, images_size = [], [], []
        for question in questions:
            question = question.replace(",", "")
            qs = random.choice(QUESTION_PARTIAL).replace("[class_name]", question)

            if self.model_config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            new_w = 336
            new_h = 336

            image_new = image.resize((new_w, new_h), Image.BILINEAR)

            image_tensor = process_images([image_new], self.image_processor, self.model_config)[0]
            images_tensor.append(image_tensor)

            input_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            input_ids.append(input_id)

            images_size.append((new_h, new_w))

        return input_ids, images_tensor, images_size, masks, image, image_name, questions

    def __len__(self):
        return len(self.dataset)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes, masks, image, image_name, questions = zip(*batch)
    return input_ids, image_tensors, image_sizes, masks, image, image_name, questions


# DataLoader
def create_data_loader(sub_dataset, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(sub_dataset, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


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

    val_dataset = ValDataset(args.image_folder, args.dataset_split)
    sub_dataset = get_chunk(val_dataset, args.num_chunks, args.chunk_idx)

    data_loader = create_data_loader(sub_dataset, tokenizer, image_processor, model.config)

    if "p16" in args.model_path:
        h, w = 16, 16
    if "p24" in args.model_path:
        h, w = 24, 24
    
    for (input_ids, image_tensor, image_sizes, masks, image, image_name, questions) in tqdm(data_loader, total=len(data_loader)):
        input_ids = input_ids[0]
        image_tensor = image_tensor[0]
        image_sizes = image_sizes[0]
        masks = masks[0]
        image = image[0]
        image_name = image_name[0]
        questions = questions[0]

        with torch.inference_mode():
            w_ori, h_ori = image.size
            predictor.set_image(np.array(image))
            for i in range(len(input_ids)):
                output_ids = model.generate(
                    input_ids[i].unsqueeze(0).to(device='cuda', non_blocking=True),
                    images=[image_tensor[i].to(dtype=torch.float16, device='cuda', non_blocking=True)],
                    image_sizes=image_sizes[i],
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
                    pred_mask = translate_sequence(mask_labels)
                except:
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
                    logits = compute_logits_from_mask(mask.to(torch.float64))
                    point_coords, point_labels = masks_sample_points(mask_upsample)

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
                gt_mask = masks[i]
                sam_mask = sam_mask[0]

                # save the pred_mask, sam_mask and gt_mask
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

                image_path = os.path.join(args.save_file, model_name, ds_split, image_name)
                if not os.path.exists(image_path):
                    os.makedirs(image_path)
                pred_mask_s.save(os.path.join(image_path, f"{i}_pred_mask.png"))
                sam_mask_s.save(os.path.join(image_path, f"{i}_sam_mask.png"))
                gt_mask_s.save(os.path.join(image_path, f"{i}_gt_mask.png"))
                image.save(os.path.join(image_path, f"{i}_image.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="./playground/data/refer_seg/")
    parser.add_argument("--sam_path", type=str, default="./llava/model/segment_anything/sam_vit_h_4b8939.pth")
    parser.add_argument("--dataset_split", type=str, default="refcoco|unc|val")
    parser.add_argument("--save_file", type=str, default="llava/eval/ref_seg_results/")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=3069)
    args = parser.parse_args()

    eval_model(args)
