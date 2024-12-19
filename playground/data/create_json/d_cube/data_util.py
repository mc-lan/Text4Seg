# -*- coding: utf-8 -*-
__author__ = "Chi Xie and Zhao Zhang"
__maintainer__ = "Chi Xie"
# data utility functions are defined in the script
import json
import pickle
import shutil

# from io import StringIO
# import string

import numpy as np
import cv2
from pycocotools import mask as cocomask

VOC_COLORMAP = [
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]


def visualize_bbox_on_image(img, bbox_list, save_path=None, thickness=3):
    img_copy = img.copy()
    for i, bbox in enumerate(bbox_list):
        color = tuple(VOC_COLORMAP[i % len(VOC_COLORMAP)])
        x, y, w, h = bbox
        img_copy = cv2.rectangle(
            img_copy, (int(x), int(y)), (int((x + w)), int(y + h)), color, thickness
        )
    if save_path:
        cv2.imwrite(save_path, img_copy)
    return img_copy


def rle2bmask(rle):
    bm = cocomask.decode(rle)
    if len(bm.shape) == 3:
        bm = np.sum(
            bm, axis=2
        )  # sometimes there are multiple binary map (corresponding to multiple segs)
    bm = bm.astype(np.uint8)  # convert to np.uint8
    return bm


def merge_rle(rle_list, is_instance=True, on_image=False):
    if is_instance:
        cm_list = []
        for rle_idx, rle in enumerate(rle_list):
            color = VOC_COLORMAP[rle_idx]
            bm = rle2bmask(rle)
            cm = cv2.cvtColor(bm, cv2.COLOR_GRAY2BGR)
            cm_list.append(cm * color)
        merge_map = np.sum(cm_list, axis=0, dtype=np.uint8)
    else:
        bm_list = [rle2bmask(rle) for rle in rle_list]
        merge_map = np.sum(bm_list, axis=0, dtype=np.uint8)
    merge_map[merge_map >= 1] = 1
    if not on_image:
        color = VOC_COLORMAP[0]
        merge_map = cv2.cvtColor(merge_map, cv2.COLOR_GRAY2BGR)
        merge_map *= np.array(color, dtype=np.uint8)

    merge_map[merge_map > 255] = 255

    if not on_image:
        tmp_sum_map = np.sum(merge_map, axis=-1)
        merge_map[tmp_sum_map == 0] = 220
    return merge_map


def merge2bin(rle_list, img_h, img_w):
    if rle_list:
        bm_list = [rle2bmask(rle) for rle in rle_list]
        merge_map = np.sum(bm_list, axis=0, dtype=np.uint8)
        merge_map[merge_map >= 1] = 255
        merge_map = np.expand_dims(merge_map, axis=-1)
        return merge_map
    else:
        return np.zeros([img_h, img_w, 1], dtype=np.uint8)


def paste_text(img, text):
    fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL
    overlay = img.copy()
    # fontFace = cv2.FONT_HERSHEY_TRIPLEX
    fontScale = 1
    thickness = 1
    backgroud_alpha = 0.8

    retval, baseLine = cv2.getTextSize(
        text, fontFace=fontFace, fontScale=fontScale, thickness=thickness
    )
    topleft = (0, 0)
    # bottomright = (topleft[0] + retval[0], topleft[1] + retval[1]+10)
    bottomright = (img.shape[1], topleft[1] + retval[1] + 10)

    cv2.rectangle(overlay, topleft, bottomright, thickness=-1, color=(250, 250, 250))
    img = cv2.addWeighted(overlay, backgroud_alpha, img, 1 - backgroud_alpha, 0)

    cv2.putText(
        img,
        text,
        (0, baseLine + 10),
        fontScale=fontScale,
        fontFace=fontFace,
        thickness=thickness,
        color=(10, 10, 10),
    )
    return img


def load_json(json_path, to_int=False):
    clean_res_dic = {}
    with open(json_path, "r", encoding="utf-8") as f_in:
        res_dic = json.load(f_in)

    for ikey, iv in res_dic.items():
        ikey = int(ikey.strip()) if to_int else ikey.strip()
        clean_res_dic[ikey] = iv

    return clean_res_dic


def path_map(src_path, obj_path):
    def inner_map(full_path):
        return full_path.replace(src_path, obj_path)


def save_pkl(src, obj_path):
    with open(obj_path, "wb") as f_out:
        pickle.dump(src, f_out)


def load_pkl(src_path):
    with open(src_path, "rb") as f_in:
        in_pkl = pickle.load(f_in)
    return in_pkl


def copy_file(src_path, obj_path):
    shutil.copy(src_path, obj_path)


def sentence_analysis():
    return 0


def add_checkerboard_bg(image, mask, save_path=None):
    # Create a new image with the same size as the original image
    new_image = np.zeros_like(image)

    # Define the size of the checkerboard pattern
    checkerboard_size = 24

    # Loop over each pixel in the mask
    for x in range(mask.shape[1]):
        for y in range(mask.shape[0]):
            # If the pixel is transparent, draw a checkerboard pattern
            if mask[y, x] == 0:
                if (x // checkerboard_size) % 2 == (y // checkerboard_size) % 2:
                    new_image[y, x] = (255, 255, 255)
                else:
                    new_image[y, x] = (128, 128, 128)
            # Otherwise, copy the corresponding pixel from the original image
            else:
                new_image[y, x] = image[y, x]

    # Save the new image with the checkerboard background
    if save_path:
        cv2.imwrite(save_path, new_image)
    return new_image


def visualize_mask_on_image(
    img, mask, save_path=None, add_edge=False, dark_background=False
):
    # Convert the mask to a binary mask if it's not already
    if mask.max() > 1:
        mask = mask.astype(np.uint8) // 255

    # Convert the mask to a 3-channel mask if it's not already
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=2)
        mask = np.tile(mask, (1, 1, 3))

    # Create a color map for the mask
    cmap = np.array([255, 117, 44], dtype=np.uint8)
    mask_colors = mask * cmap

    # Add an opaque white edge to the mask if desired
    if add_edge:
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=2)
            mask = np.tile(mask, (1, 1, 3))

        kernel = np.ones((5, 5), dtype=np.uint8)
        mask_edge = cv2.erode(mask, kernel, iterations=1)
        mask_edge = mask - mask_edge

        # mask_edge = np.tile(mask_edge[:, :, np.newaxis], [1, 1, 3])
        mask_colors[mask_edge > 0] = 255

    # Overlay the mask on the masked image
    if dark_background:
        masked_img = cv2.addWeighted(img, 0.4, mask_colors, 0.6, 0)
    else:
        masked_img = img.copy()
        masked_img[mask > 0] = cv2.addWeighted(img, 0.4, mask_colors, 0.6, 0)[mask > 0]

    # Save the result to the specified path if provided
    if save_path is not None:
        cv2.imwrite(save_path, masked_img)

    return masked_img


# def visualize_mask_on_image(img, mask, save_path=None, add_edge=False):
#     # Convert the mask to a binary mask if it's not already
#     if mask.max() > 1:
#         mask = mask.astype(np.uint8) // 255

#     # Convert the mask to a 3-channel mask if it's not already
#     if len(mask.shape) == 2:
#         mask = np.expand_dims(mask, axis=2)
#         mask = np.tile(mask, (1, 1, 3))

#     # Create a color map for the mask
#     cmap = np.array([255, 117, 44], dtype=np.uint8)
#     mask_colors = mask * cmap

#     # Add an opaque white edge to the mask if desired
#     if add_edge:
#         if len(mask.shape) == 2:
#             mask = np.expand_dims(mask, axis=2)
#             mask = np.tile(mask, (1, 1, 3))

#         kernel = np.ones((5, 5), dtype=np.uint8)
#         mask_edge = cv2.erode(mask, kernel, iterations=1)
#         mask_edge = mask - mask_edge

#         # mask_edge = np.tile(mask_edge[:, :, np.newaxis], [1, 1, 3])
#         mask_colors[mask_edge > 0] = 255

#     # Overlay the mask on the masked image
#     masked_img = cv2.addWeighted(img, 0.5, mask_colors, 0.5, 0)

#     # Save the result to the specified path if provided
#     if save_path is not None:
#         cv2.imwrite(save_path, masked_img)

#     return masked_img
