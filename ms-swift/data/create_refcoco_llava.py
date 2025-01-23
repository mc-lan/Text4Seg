import os
import numpy as np
from PIL import Image
import random
from pycocotools import mask
from refer import REFER
from question_answer_list import QUESTION_ALL, QUESTION_PARTIAL, QUESTION_CONDITION, ANSWER_ALL, ANSWER_PARTIAL, \
    ANSWER_CONDITION
from utils import encode_mask


w_new = h_new = 16
data_path = "./datasets/refer_seg"
dss = ["refcoco", "refcoco+", "refcocog", "refclef"]
Content = []
j = 0

for ds in dss:
    if ds == "refcocog":
        splitBy = "umd"
    else:
        splitBy = "unc"

    refer_api = REFER(data_path, ds, splitBy)

    ref_ids_train = refer_api.getRefIds(split="train")
    images_ids_train = refer_api.getImgIds(ref_ids=ref_ids_train)  
    loaded_images = refer_api.loadImgs(image_ids=images_ids_train) 

    for _ in range(2):
        for num, image_info in zip(range(len(loaded_images)), loaded_images):
            refs = refer_api.imgToRefs[image_info["id"]]
            for ref in refs:
                sentences = ref['sentences']
                ann = refer_api.refToAnn[ref['ref_id']]

                if type(ann["segmentation"][0]) == list:  # polygon
                    rle = mask.frPyObjects(
                        ann["segmentation"],
                        image_info["height"],
                        image_info["width"],
                    )
                else:
                    rle = ann["segmentation"]
                    for i in range(len(rle)):
                        if not isinstance(rle[i]["counts"], bytes):
                            rle[i]["counts"] = rle[i]["counts"].encode()
                m = mask.decode(rle)
                m = np.sum(m, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
                m[m > 1] = 1

                annotation = Image.fromarray(m.astype(np.uint8))
                annotation = annotation.resize((w_new, h_new), Image.NEAREST)  
                array_annotation = np.array(annotation)  
                array_annotation = array_annotation.flatten()

                for sentence in sentences:
                    item = {}
                    question = random.choice(QUESTION_PARTIAL).replace("[class_name]", sentence['sent'])

                    item["query"] = f"<image>{question}"

                    label_each_pixel = [sentence['sent'] if label > 0 else "others" for label in array_annotation]

                    ####  index  ####
                    label_each_pixel = np.reshape(label_each_pixel, (h_new, w_new))
                    SEG = encode_mask(label_each_pixel)
                    ####  index  ####

                    answer = random.choice(ANSWER_PARTIAL).replace("[class_name]", sentence['sent']) + f"\n<seg>{SEG}</seg>"

                    item["response"] = f"{answer}"

                    if ds == "refclef":
                        item["images"] = [os.path.join("./datasets", "refer_seg/images/saiapr_tc-12", image_info["file_name"])]
                    else:
                        item["images"] = [os.path.join("./datasets", "refer_seg/images/coco_2014/train2014", image_info["file_name"])]

                    Content.append(item)
                    j += 1
                    print(j)

import json

with open("./datasets/json_files/refcoco_16_2_llava.json", "w") as f:
    json.dump(Content, f, indent=4)

