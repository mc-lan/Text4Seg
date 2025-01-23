import os
import numpy as np
from PIL import Image
import random
from pycocotools.coco import COCO
from question_answer_list import QUESTION_PARTIAL, ANSWER_PARTIAL, ANSWER_CONDITION, QUESTION_OBJECT_ALL, ANSWER_OBJECT_ALL, QUESTION_OBJECT_PART, ANSWER_OBJECT_PART
from utils import encode_mask


def init_pascal_part(base_image_dir):
    coco_api_pascal_part = COCO(
        os.path.join(base_image_dir, "vlpart", "pascal_part", "train.json")
    )
    all_classes = coco_api_pascal_part.loadCats(coco_api_pascal_part.getCatIds())
    class_map_pascal_part = {}
    for cat in all_classes:
        cat_main, cat_part = cat["name"].strip().split(":")
        name = (cat_main, cat_part)
        class_map_pascal_part[cat["id"]] = name
    img_ids = coco_api_pascal_part.getImgIds()
    print("pascal_part: ", len(img_ids))
    return class_map_pascal_part, img_ids, coco_api_pascal_part

class_map_pascal_part, img_ids, coco_api_pascal_part = init_pascal_part("./playground/data")

labels = {}
for key, value in class_map_pascal_part.items():
    obj, part = value
    labels[key] = part

h_new = w_new = 32
Content = []
i = 0
for num, img_id in zip(range(len(img_ids)), img_ids):
    image_info = coco_api_pascal_part.loadImgs(img_id)[0]
    file_name = image_info["file_name"].split("/")[-1]

    item = {}
    item["id"] = file_name.split(".")[0]
    item["image"] = "vlpart/pascal_part/VOCdevkit/VOC2010/JPEGImages/" + file_name

    annIds = coco_api_pascal_part.getAnnIds(imgIds=image_info["id"])
    anns = coco_api_pascal_part.loadAnns(annIds)

    annotation = np.ones((image_info["height"], image_info["width"])) * -1
    for ann in anns:
        mask = coco_api_pascal_part.annToMask(ann) * ann["category_id"]
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

    # remove -1 from labels_in_image
    labels_in_image = labels_in_image[labels_in_image != -1]
    # find the labels not in the image from class_map_paco_lvis
    labels_not_in_image = [label for label in class_map_pascal_part.keys() if label not in labels_in_image]

    # conversations
    conversation_list = []

    question_all = False

    object_in_image = [class_map_pascal_part[label_in_image][0] for label_in_image in labels_in_image]
    object_in_image = set(object_in_image)

    if len(object_in_image) == 0:
        continue

    for round in range(2):
        random_number = random.random()

        if question_all:
            if random.random() < 0.3:
                random_number = 0.55
            else:
                random_number = 0.9

        if random_number < 0.5:
            question_all = True
            select_objects = random.sample(object_in_image, random.randint(1, len(object_in_image)))


            question = random.choice(QUESTION_OBJECT_ALL).replace('[object]', '| '.join(select_objects))
            if round == 0:
                conversation_list.append({"from": "human", "value": f"<image>\n{question}"})
            else:
                conversation_list.append({"from": "human", "value": f"{question}"})

            # translate the labels to categories
            label_each_pixel_diversity = []
            for label in array_annotation:
                if label in labels:
                    obj, part = class_map_pascal_part[label]
                    if obj in select_objects:
                        label_each_pixel_diversity.append(part)
                    else:
                        label_each_pixel_diversity.append("others")
                else:
                    label_each_pixel_diversity.append("others")

            ####  index  ####
            label_each_pixel_diversity = np.reshape(label_each_pixel_diversity, (h_new, w_new))
            SEG = encode_mask(label_each_pixel_diversity)
            ####  index  ####

            answer = random.choice(ANSWER_OBJECT_ALL).replace('[object]', '| '.join(select_objects)) + f"\n<seg>{SEG}</seg>"

            conversation_list.append({"from": "gpt", "value": answer})

        elif random_number < 0.6:
            select_objects = random.sample(object_in_image, random.randint(1, len(object_in_image)))
            random.shuffle(select_objects)

            question = random.choice(QUESTION_PARTIAL).replace('[class_name]', '| '.join(select_objects))
            if round == 0:
                conversation_list.append({"from": "human", "value": f"<image>\n{question}"})
            else:
                conversation_list.append({"from": "human", "value": f"{question}"})

            # translate the labels to categories
            label_each_pixel_diversity = []
            for label in array_annotation:
                if label in labels:
                    obj, part = class_map_pascal_part[label]
                    if obj in select_objects:
                        label_each_pixel_diversity.append(obj)
                    else:
                        label_each_pixel_diversity.append("others")
                else:
                    label_each_pixel_diversity.append("others")

            ####  index  ####
            label_each_pixel_diversity = np.reshape(label_each_pixel_diversity, (h_new, w_new))
            SEG = encode_mask(label_each_pixel_diversity)
            ####  index  ####

            answer = random.choice(ANSWER_PARTIAL).replace('[class_name]',
                                                              '| '.join(select_objects)) + f"\n<seg>{SEG}</seg>"

            conversation_list.append({"from": "gpt", "value": answer})

        else:
            select_objects = random.choice(list(object_in_image))
            all_parts = [class_map_pascal_part[label][1] for label in class_map_pascal_part.keys() if
                         class_map_pascal_part[label][0] == select_objects]
            exist_parts = [class_map_pascal_part[label_in_image][1] for label_in_image in labels_in_image if
                           class_map_pascal_part[label_in_image][0] == select_objects]
            not_exist_parts = list(set(all_parts) - set(exist_parts))

            # random select several categories in labels_in_image
            if random.random() < 0.8:
                parts = random.sample(exist_parts, 1)
            else:
                parts = random.sample(exist_parts, random.randint(1, min(len(exist_parts), 4)))

            # random add some parts not in the image
            if random.random() < 0.2 and len(not_exist_parts) > 0:
                parts += random.sample(not_exist_parts, random.randint(1, min(len(not_exist_parts), 2)))

            random.shuffle(parts)

            question = random.choice(QUESTION_OBJECT_PART).replace('[object]', select_objects).replace('[part]', '| '.join(parts))
            if round == 0:
                conversation_list.append({"from": "human", "value": f"<image>\n{question}"})
            else:
                conversation_list.append({"from": "human", "value": f"{question}"})

            # translate the labels to categories
            label_each_pixel_partial = []
            for label in array_annotation:
                if label in labels:
                    obj, part = class_map_pascal_part[label]
                    if obj in select_objects and part in parts:
                        label_each_pixel_partial.append(part)
                    else:
                        label_each_pixel_partial.append("others")
                else:
                    label_each_pixel_partial.append("others")

            ####  index  ####
            label_each_pixel_partial = np.reshape(label_each_pixel_partial, (h_new, w_new))
            SEG = encode_mask(label_each_pixel_partial)
            ####  index  ####

            answer = random.choice(ANSWER_OBJECT_PART).replace('[object]', select_objects).replace('[part]', '| '.join(parts)) + f"\n<seg>{SEG}</seg>"

            conversation_list.append({"from": "gpt", "value": answer})

        item["conversations"] = conversation_list

    Content.append(item)
    i += 1
    print(i)
    # if i > 10:
    #     break

import json
with open("./playground/data/json_files/pascal_part_32_f_three_round.json", "w") as f:
    json.dump(Content, f, indent=4)

