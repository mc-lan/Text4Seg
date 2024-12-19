import os
import json
import argparse

import numpy as np
import torch
from PIL import Image
from llava.eval.utils import AverageMeter, Summary, intersectionAndUnionGPU

intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
acc_iou_meter = AverageMeter("mIoU", ":6.3f", Summary.SUM)

intersection_meter_sam = AverageMeter("Intersec", ":6.3f", Summary.SUM)
union_meter_sam = AverageMeter("Union", ":6.3f", Summary.SUM)
acc_iou_meter_sam = AverageMeter("mIoU", ":6.3f", Summary.SUM)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_split", type=str, default="PAS20")
    parser.add_argument("--save_file", type=str, default="llava/eval/semantic_seg_results/llava-v1.5-7b-lora-r64-coco5")
    args = parser.parse_args()
    
    
    if args.dataset_split == "PAS20":
        K_classes = 21
    elif args.dataset_split == "PC59":
        K_classes = 60
    elif args.dataset_split == "ADE20K":
        K_classes = 151
    else:
        K_classes = 459

    image_path = os.path.join(args.save_file, args.dataset_split)

    image_files = os.listdir(image_path)
    ii = 0
    for image_file in image_files:
        ii = ii+1
        print(ii)
        image_file_path = os.path.join(image_path, image_file)
        image = Image.open(os.path.join(image_file_path, f"image.png"))
        pred_mask = Image.open(os.path.join(image_file_path, f"pred_mask.png"))
        gt_mask = Image.open(os.path.join(image_file_path, f"gt_mask.png"))

        sam_mask = Image.open(os.path.join(image_file_path, f"sam_mask.png"))
        sam_mask = torch.from_numpy(np.array(sam_mask)).to(dtype=torch.float32)
        
        pred_mask = torch.from_numpy(np.array(pred_mask)).to(dtype=torch.float32)
        gt_mask = torch.from_numpy(np.array(gt_mask)).to(dtype=torch.float32)

        intersection, union, _ = intersectionAndUnionGPU(pred_mask.contiguous().clone(), gt_mask.contiguous(), K_classes, ignore_index=0)
        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()

        intersection_sam, union_sam, _ = intersectionAndUnionGPU(sam_mask.contiguous().clone(), gt_mask.contiguous(), K_classes, ignore_index=0)
        intersection_sam, union_sam = intersection_sam.cpu().numpy(), union_sam.cpu().numpy()

        intersection_meter.update(intersection)
        union_meter.update(union)

        intersection_meter_sam.update(intersection_sam)
        union_meter_sam.update(union_sam)
    
    # Calculate the mIoU across all classes
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)  # Adding small value to avoid division by zero
    miou = np.mean(iou_class)  # Mean of IoU across all classes
    print("llava_mask: mIoU: {:.4f}".format(miou))

    iou_class = intersection_meter_sam.sum / (union_meter_sam.sum + 1e-10)  # Adding small value to avoid division by zero
    miou = np.mean(iou_class)  # Mean of IoU across all classes
    print("sam_mask: mIoU: {:.4f}".format(miou))