import os
import numpy as np
from PIL import Image
import random
from pycocotools.coco import COCO
from question_answer_list import QUESTION_ALL, QUESTION_PARTIAL, QUESTION_CONDITION, ANSWER_ALL, ANSWER_PARTIAL, ANSWER_CONDITION
from utils import encode_mask


def init_paco_lvis(base_image_dir):
    coco_api_paco_lvis = COCO(
        os.path.join(
            base_image_dir, "vlpart", "paco_lvis", "annotations", "paco_lvis_v1_train.json"
        )
    )
    all_classes = coco_api_paco_lvis.loadCats(coco_api_paco_lvis.getCatIds())
    class_map_paco_lvis = {}
    for cat in all_classes:
        cat_split = cat["name"].strip().split(":")
        if len(cat_split) == 1:
            name = cat_split[0].split("_(")[0]
        else:
            assert len(cat_split) == 2
            obj, part = cat_split
            obj = obj.split("_(")[0]
            part = part.split("_(")[0]
            name = (obj, part)
        class_map_paco_lvis[cat["id"]] = name
    img_ids = coco_api_paco_lvis.getImgIds()
    print("paco_lvis: ", len(img_ids))
    return class_map_paco_lvis, img_ids, coco_api_paco_lvis

class_map_paco_lvis, img_ids, coco_api_paco_lvis = init_paco_lvis("./playground/data")

labels = {}
for key, value in class_map_paco_lvis.items():
    if isinstance(value, tuple):
        obj, part = value
        if random.random() < 0.3:
            name = obj + " " + part
        else:
            name = "{} of the {}".format(part, obj)
    else:
        name = value
    labels[key] = name

h_new = w_new = 32
Content = []
i = 0
for num, img_id in zip(range(len(img_ids)), img_ids):
    image_info = coco_api_paco_lvis.loadImgs(img_id)[0]
    file_name = image_info["file_name"].split("/")[-1]

    item = {}
    item["id"] = file_name.split(".")[0]
    item["image"] = "coco/train2017/" + file_name

    annIds = coco_api_paco_lvis.getAnnIds(imgIds=image_info["id"])
    anns = coco_api_paco_lvis.loadAnns(annIds)

    annotation = np.ones((image_info["height"], image_info["width"])) * -1
    for ann in anns:
        mask = coco_api_paco_lvis.annToMask(ann) * ann["category_id"]
        annotation[mask > 0] = mask[mask > 0]

    annotation = Image.fromarray(annotation.astype(np.int16))
    item['height'] = h_new
    item['width'] = w_new

    annotation = annotation.resize((w_new, h_new), Image.NEAREST)  
    array_annotation = np.array(annotation) 
    array_annotation = array_annotation.flatten()

    # translate the labels to categories: exclude -1
    label_each_pixel = [labels[label] if label in labels else "others" for label in array_annotation]

    labels_in_image = np.unique(array_annotation)

    if len(labels_in_image) == 1:
        continue

    # remove -1 from labels_in_image
    labels_in_image = labels_in_image[labels_in_image != -1]
    # find the labels not in the image from class_map_paco_lvis
    labels_not_in_image = [label for label in class_map_paco_lvis.keys() if label not in labels_in_image]

    # conversations
    conversation_list = []

    question_all = False

    for round in range(2):
        # select one quesetion from questions_all or questions_partial
        question = random.choice(QUESTION_PARTIAL)

        # random select several categories in labels_in_image
        if random.random() < 0.8:
            partial_labels = random.sample(labels_in_image.tolist(), 1)
        else:
            partial_labels = random.sample(labels_in_image.tolist(), random.randint(1, min(len(labels_in_image), 3)))

        # translate the labels to categories
        partial_labels = [labels[partial_label] for partial_label in partial_labels]

        random.shuffle(partial_labels)

        question = question.replace("[class_name]", "| ".join(partial_labels))

        if round == 0:
            conversation_list.append({"from": "human", "value": f"<image>\n{question}"})
        else:
            conversation_list.append({"from": "human", "value": f"{question}"})

        label_each_pixel_partial = [label if label in partial_labels else "others" for label in label_each_pixel]

        ####  index  ####
        label_each_pixel_partial = np.reshape(label_each_pixel_partial, (h_new, w_new))
        SEG = encode_mask(label_each_pixel_partial)
        ####  index  ####

        answer = random.choice(ANSWER_PARTIAL).replace("[class_name]", "| ".join(partial_labels)) + f"\n<seg>{SEG}</seg>"

        conversation_list.append({"from": "gpt", "value": answer})

    item["conversations"] = conversation_list

    Content.append(item)
    i += 1
    print(i)
    # if i > 1000:
    #     break

import json
with open("./playground/data/json_files/paco_lvis_32_f_two_round.json", "w") as f:
    json.dump(Content, f, indent=4)

