import os
import numpy as np
from PIL import Image
import random
from pycocotools import mask
from pycocotools.coco import COCO
from d_cube import D3
import matplotlib.pyplot as plt
from question_answer_list import QUESTION_ALL, QUESTION_PARTIAL, QUESTION_CONDITION, ANSWER_ALL, ANSWER_PARTIAL, \
    ANSWER_CONDITION

RESOLUTION = 336

def encode_mask(mask_list):
    rows = []
    for row in mask_list:
        encoded_row = []
        count = 1
        for j in range(1, len(row)):
            if row[j] == row[j - 1]:
                count += 1
            else:
                encoded_row.append(f"{row[j - 1]} *{count}")
                count = 1
        encoded_row.append(f"{row[-1]} *{count}")
        rows.append("| ".join(encoded_row))
    return "\n ".join(rows) + "\n"


data_path = "./playground/data/refer_seg"

IMG_ROOT = "./playground/data/refer_seg/d_cube/d3_images"
PKL_ANNO_PATH = "./playground/data/refer_seg/d_cube/d3_pkl"

d3 = D3(IMG_ROOT, PKL_ANNO_PATH)
ds = 'd_cube'
 # get the annotation ids in the dataset
all_img_info = d3.load_imgs(d3.get_img_ids())  # load images by passing a list containing some image ids
all_anno_info = d3.load_annos(d3.get_anno_ids())  # load annotations by passing a list containing some annotation ids
all_sent_info = d3.load_sents(d3.get_sent_ids())  # load sentences by passing a list containing some sentence ids

Content = []
j = 0
for _ in range(2):
    for num, anno_info in zip(range(len(all_anno_info)), all_anno_info):
        item = {}

        img_id = anno_info["image_id"]
        image_info = all_img_info[img_id]
        item["id"] = image_info["id"]
        item["image"] = os.path.join("refer_seg/d_cube/d3_images", image_info["file_name"])

        sentences = all_sent_info[anno_info["sent_id"][0]-1]
        mask = d3.get_mask([anno_info])['mask']

        annotation = Image.fromarray(mask.astype(np.uint8))

        # resize the image, the longest side is resolution, keep the aspect ratio
        w, h = annotation.size

        # resolution = random.choice([RESOLUTION, RESOLUTION-14, RESOLUTION-28])
        # if w < h:
        #     if h < resolution:
        #         new_h, new_w = h, w
        #     else:
        #         new_h = resolution
        #         new_w = int(w / h * new_h)
        # else:
        #     if w < resolution:
        #         new_h, new_w = h, w
        #     else:
        #         new_w = resolution
        #         new_h = int(h / w * new_w)

        resolution = RESOLUTION
        if w < h:
            new_h = resolution
            new_w = int(w / h * new_h)
        else:
            new_w = resolution
            new_h = int(h / w * new_w)

        w_new = new_w // 14
        h_new = new_h // 14

        w_new = 16
        h_new = 16

        item['height'] = h_new
        item['width'] = w_new

        annotation = annotation.resize((w_new, h_new), Image.NEAREST)  # 32x21
        array_annotation = np.array(annotation).flatten()  # 21 x 32

        # conversations
        conversation_list = []

        # select one quesetion from questions_all or questions_partial
        question = random.choice(QUESTION_PARTIAL).replace("[class_name]", sentences['raw_sent'])

        conversation_list.append({"from": "human", "value": f"<image>\n{question}"})

        label_each_pixel = [sentences['raw_sent'] if label > 0 else "others" for label in array_annotation]

        ####  index  ####
        label_each_pixel = np.reshape(label_each_pixel, (h_new, w_new))
        SEG = encode_mask(label_each_pixel)
        ####  index  ####

        answer = random.choice(ANSWER_PARTIAL).replace("[class_name]", sentences['raw_sent']) + f"\n<seg>{SEG}</seg>"

        conversation_list.append({"from": "gpt", "value": answer})

        item["conversations"] = conversation_list

        Content.append(item)
        j += 1
        print(j)
        # if j > 10:
        #     break

import json
with open("./playground/data/json_files/" + ds +"_16_two_round_2.json", "w") as f:
    json.dump(Content, f, indent=4)

