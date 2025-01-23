import os
import numpy as np
from PIL import Image
import random
from pycocotools import mask
from grefer import G_REFER
from question_answer_list import QUESTION_ALL, QUESTION_PARTIAL, QUESTION_CONDITION, ANSWER_ALL, ANSWER_PARTIAL, \
    ANSWER_CONDITION
from utils import encode_mask


data_path = "./datasets/refer_seg"
ds = "grefcoco"
splitBy = "unc"
refer_api = G_REFER(data_path, ds, splitBy)
w_new = h_new = 16

ref_ids_train = refer_api.getRefIds(split="train")
images_ids_train = refer_api.getImgIds(ref_ids=ref_ids_train)  
loaded_images = refer_api.loadImgs(image_ids=images_ids_train) 

Content = []
j = 0
num_FAILD = 0
for _ in range(2):
    for num, image_info in zip(range(len(loaded_images)), loaded_images):
        refs = refer_api.imgToRefs[image_info["id"]]
        for ref in refs:
            if ds == "refclef":
                img_path = os.path.join("./datasets", "refer_seg/images/saiapr_tc-12", image_info["file_name"])
            else:
                img_path = os.path.join("./datasets", "refer_seg/images/coco_2014/train2014", image_info["file_name"])

            sentences = ref['sentences']
            anns = refer_api.refToAnn[ref['ref_id']]

            if None in anns or "segmentation" not in anns[0]:
                m = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
            else:
                m = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
                for ann in anns:
                    if type(ann["segmentation"]) == list and type(ann["segmentation"][0]) == list:  # polygon
                        rle = mask.frPyObjects(
                            ann["segmentation"],
                            image_info["height"],
                            image_info["width"],
                        )
                    else:
                        num_FAILD += 1
                        pass
                        # rle = ann["segmentation"]
                        # for i in range(len(rle["counts"])):
                        #     if not isinstance(rle["counts"][i], bytes):
                        #         rle["counts"][i] = rle[i]["counts"][i].encode()
                    m = m + np.sum(mask.decode(rle), axis=2)
                m[m > 1] = 1
            annotation = Image.fromarray(m.astype(np.uint8))
            annotation = annotation.resize((w_new, h_new), Image.NEAREST)  
            array_annotation = np.array(annotation) 
            array_annotation = array_annotation.flatten()

            for sentence in sentences:
                item = {}
                # conversations
                conversation_list = []
                question = random.choice(QUESTION_PARTIAL).replace("[class_name]", sentence['sent'])

                conversation_list.append({"from": "user", "value": f"<img>{img_path}</img>{question}"})

                label_each_pixel = [sentence['sent'] if label > 0 else "others" for label in array_annotation]

                ####  index  ####
                label_each_pixel = np.reshape(label_each_pixel, (h_new, w_new))
                SEG = encode_mask(label_each_pixel)
                ####  index  ####

                answer = random.choice(ANSWER_PARTIAL).replace("[class_name]", sentence['sent']) + f"\n<seg>{SEG}</seg>"

                conversation_list.append({"from": "assistant", "value": answer})

                item["conversations"] = conversation_list

                Content.append(item)
                j += 1
                print(j)

import json

with open("./datasets/json_files/grefcoco_16_2_qwen.json", "w") as f:
    json.dump(Content, f, indent=4)

