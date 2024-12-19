import os
import json
import argparse
import datetime

import numpy as np
import torch
from PIL import Image
from text4seg.utils import AverageMeter, Summary, intersectionAndUnionGPU, masks_to_boxes, box_iou
# from text4seg.utils import crf_refine
# from cv2 import imread, imwrite

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_split", type=str, default="refcoco|unc|val")
    parser.add_argument("--save_file", type=str, default="llava/eval/ref_seg_results/")
    args = parser.parse_args()

    ds_split = args.dataset_split.replace("|", "_")

    image_path = os.path.join(args.save_file, ds_split)
    image_files = os.listdir(image_path)

    ALL_samples = 0
    Correct_samples_mllm = 0
    Correct_samples_sam = 0

    for image_file in image_files:
        image_file_path = os.path.join(image_path, image_file)
        images_num = len(os.listdir(image_file_path))
        num = images_num // 4
        intersection, union, acc_iou = 0.0, 0.0, 0.0
        for i in range(num):
            ALL_samples += 1

            pred_mask = Image.open(os.path.join(image_file_path, f"{i}_pred_mask.png"))
            sam_mask = Image.open(os.path.join(image_file_path, f"{i}_sam_mask.png"))
            gt_mask = Image.open(os.path.join(image_file_path, f"{i}_gt_mask.png"))

            # img = imread(os.path.join(image_file_path, f"{i}_image.png"))
            # pred_mask = crf_refine(img, np.array(pred_mask)//255)
            # pred_mask = torch.from_numpy(pred_mask).to(dtype=torch.float32)

            pred_mask = torch.from_numpy(np.array(pred_mask)//255).to(dtype=torch.float32)
            sam_mask = torch.from_numpy(np.array(sam_mask)//255).to(dtype=torch.float32)
            gt_mask = torch.from_numpy(np.array(gt_mask)//255).to(dtype=torch.float32)

            pred_box = masks_to_boxes(pred_mask.unsqueeze(0))
            sam_box = masks_to_boxes(sam_mask.unsqueeze(0))
            gt_box =  masks_to_boxes(gt_mask.unsqueeze(0))

            pred_iou, union = box_iou(pred_box, gt_box)
            sam_iou, union = box_iou(sam_box, gt_box)

            if pred_iou > 0.5:
                Correct_samples_mllm += 1

            if sam_iou > 0.5:
                Correct_samples_sam += 1

    mllm_acc = Correct_samples_mllm / ALL_samples * 100
    sam_acc = Correct_samples_sam / ALL_samples * 100


    current_time = "\n" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": "
    file_text = args.save_file + ": " + args.dataset_split + "  "
    output_text = "mllm_mask:  acc: {:.4f}".format(mllm_acc) + "  " + "sam_mask:  acc: {:.4f}".format(sam_acc)

    with open("output_eval/output.txt", "a") as file:
        file.write(current_time)
        file.write(file_text)
        file.write(output_text)
