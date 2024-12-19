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

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type, inference_stream
)
from swift.utils import seed_everything


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
    
    image = Image.open(args.image_path).convert('RGB')

    question = args.query

    h = w = args.visual_tokens

    with torch.inference_mode():
        w_ori, h_ori = image.size
        predictor.set_image(np.array(image))
        
        if "qwen" in args.model_type:
            response, _ = inference(model, template, question)
        else:
            response, _ = inference(model, template, question, images=image)
        outputs = response.strip()

        # get context between <seg> and </seg>
        mask_labels = outputs.split("<seg>")[1].split("</seg>")[0]
        mask_labels = decode_mask(mask_labels)
        pred_mask = translate_sequence(mask_labels)

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

        sam_mask = sam_mask[0]

        # # save the pred_mask, sam_mask and gt_mask
        pred_mask_s = mask_upsample.cpu().numpy().astype("uint8") * 255
        sam_mask_s = sam_mask.astype("uint8") * 255
        
        pred_mask_s = Image.fromarray(pred_mask_s).convert('L')
        sam_mask_s = Image.fromarray(sam_mask_s).convert('L')
        
        # save the pred_mask, sam_mask and gt_mask
        pred_mask_s.save(f"{args.save_file}/pred_mask.png")
        sam_mask_s.save(f"{args.save_file}/sam_mask.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="internvl2-8b")
    parser.add_argument("--model_id_or_path", type=str, default="output_checkpoints/refcoco_5e_lr2e-4_bs128_r64_16/internvl2-8b/v0-20240910-021121/checkpoint-33500-merged")
    parser.add_argument("--image-path", type=str, default="./text4seg/dog-cat.jpg")
    parser.add_argument("--sam_path", type=str, default="./sam_vit_h_4b8939.pth")
    parser.add_argument("--query", type=str, default="Please segment only the black dog in the image.")
    parser.add_argument("--save_file", type=str, default="output_eval/demo/")
    parser.add_argument("--visual_tokens", type=int, default=16)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()

    eval_model(args)
