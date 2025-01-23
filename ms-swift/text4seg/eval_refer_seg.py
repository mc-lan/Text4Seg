import os
import json
import argparse
import datetime

import numpy as np
import torch
from PIL import Image
from text4seg.utils import AverageMeter, Summary, intersectionAndUnionGPU
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

    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    for image_file in image_files:
        image_file_path = os.path.join(image_path, image_file)
        images_num = len(os.listdir(image_file_path))
        num = images_num // 4
        intersection, union, acc_iou = 0.0, 0.0, 0.0
        for i in range(num):
            pred_mask = Image.open(os.path.join(image_file_path, f"{i}_pred_mask.png"))
            gt_mask = Image.open(os.path.join(image_file_path, f"{i}_gt_mask.png"))

            # img = imread(os.path.join(image_file_path, f"{i}_image.png"))
            # pred_mask = crf_refine(img, np.array(pred_mask)//255)
            # pred_mask = torch.from_numpy(pred_mask).to(dtype=torch.float32)

            pred_mask = torch.from_numpy(np.array(pred_mask)//255).to(dtype=torch.float32)
            gt_mask = torch.from_numpy(np.array(gt_mask)//255).to(dtype=torch.float32)

            intersection_i, union_i, _ = intersectionAndUnionGPU(
                pred_mask.contiguous().clone(), gt_mask.contiguous(), 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0

        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy() / num
        intersection_meter.update(intersection)
        union_meter.update(union)
        acc_iou_meter.update(acc_iou, n=num)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]

    intersection_meter_sam = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter_sam = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter_sam = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    for image_file in image_files:
        image_file_path = os.path.join(image_path, image_file)
        images_num = len(os.listdir(image_file_path))
        num = images_num // 4
        intersection, union, acc_iou = 0.0, 0.0, 0.0
        for i in range(num):
            sam_mask = Image.open(os.path.join(image_file_path, f"{i}_sam_mask.png"))
            gt_mask = Image.open(os.path.join(image_file_path, f"{i}_gt_mask.png"))

            sam_mask = torch.from_numpy(np.array(sam_mask)//255).to(dtype=torch.float32)
            gt_mask = torch.from_numpy(np.array(gt_mask)//255).to(dtype=torch.float32)


            intersection_i, union_i, _ = intersectionAndUnionGPU(
                sam_mask.contiguous().clone(), gt_mask.contiguous(), 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0

        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy() / num
        intersection_meter_sam.update(intersection)
        union_meter_sam.update(union)
        acc_iou_meter_sam.update(acc_iou, n=num)

    iou_class = intersection_meter_sam.sum / (union_meter_sam.sum + 1e-10)
    ciou_sam = iou_class[1]
    giou_sam = acc_iou_meter_sam.avg[1]

    current_time = "\n" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": "
    file_text = args.save_file + ": " + args.dataset_split + "  "
    output_text = "mllm_mask:  giou: {:.4f}, ciou: {:.4f}".format(giou, ciou) + "  " + "sam_mask:  giou: {:.4f}, ciou: {:.4f}".format(giou_sam, ciou_sam)

    with open("output_eval/output.txt", "a") as file:
        file.write(current_time)
        file.write(file_text)
        file.write(output_text)
