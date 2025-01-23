import os
import numpy as np
from PIL import Image
import random
import glob
import json
from question_answer_list import QUESTION_ALL, QUESTION_PARTIAL, QUESTION_CONDITION, ANSWER_ALL, ANSWER_PARTIAL, ANSWER_CONDITION

RESOLUTION = 448

def encode_mask(mask_list):
    rows = []
    for row in mask_list:
        encoded_row = []
        count = 1
        for j in range(1, len(row)):
            if row[j] == row[j-1]:
                count += 1
            else:
                encoded_row.append(f"{row[j-1]} *{count}")
                count = 1
        encoded_row.append(f"{row[-1]} *{count}")
        rows.append(", ".join(encoded_row))
    return "; ".join(rows) + ";"



def init_mapillary(base_image_dir):
    mapillary_data_root = os.path.join(base_image_dir, "mapillary")
    with open(os.path.join(mapillary_data_root, "config_v2.0.json")) as f:
        mapillary_classes = json.load(f)["labels"]
    mapillary_classes = [x["readable"].lower() for x in mapillary_classes]
    mapillary_classes = np.array(mapillary_classes)
    mapillary_labels = sorted(
        glob.glob(
            os.path.join(mapillary_data_root, "training", "v2.0", "labels", "*.png")
        )
    )
    mapillary_images = [
        x.replace(".png", ".jpg").replace("v2.0/labels", "images")
        for x in mapillary_labels
    ]
    print("mapillary: ", len(mapillary_images))
    return mapillary_classes, mapillary_images, mapillary_labels

mapillary_classes, mapillary_images, mapillary_labels = init_mapillary("/data/mclan/LLaVa/playground/data")


# read cls_coco_stuff.txt to get the labels
labels = {}
with open("cls_mapillary.txt") as f:
    for idx, line in enumerate(f):
        labels[idx] = line.strip()

Content = []
i = 0
for num, image, label in zip(range(len(mapillary_images)), mapillary_images, mapillary_labels):
    item = {}
    image_name = image.split("/")[-1]
    item["id"] = image_name.split(".")[0]
    item["image"] = "maplillary/training/images/" + image_name

    annotation = Image.open(label)

    # resize the image, the longest side is resolution, keep the aspect ratio
    w, h = annotation.size
    resolution = random.choice([RESOLUTION, RESOLUTION-14, RESOLUTION-28, RESOLUTION-42, RESOLUTION-56])
    if w < h:
        if h < resolution:
            new_h, new_w = h, w
        else:
            new_h = resolution
            new_w = int(w / h * new_h)
    else:
        if w < resolution:
            new_h, new_w = h, w
        else:
            new_w = resolution
            new_h = int(h / w * new_w)

    w_new = new_w // 14
    h_new = new_h // 14

    item['height'] = h_new
    item['width'] = w_new

    annotation = annotation.resize((w_new, h_new), Image.NEAREST)  # 32x21
    array_annotation = np.array(annotation)  # 21 x 32
    array_annotation = array_annotation.flatten()

    # translate the labels to categories:
    label_each_pixel = [labels[label] for label in array_annotation]

    labels_in_image = np.unique(array_annotation)

    if len(labels_in_image) == 1:
        continue

    # remove 255 from labels_in_image
    labels_not_in_image = [label for label in range(123) if label not in labels_in_image]

    # conversations
    conversation_list = []

    question_all = False

    for round in range(random.randint(1, 2)):
        # select one quesetion from questions_all or questions_partial
        random_number = random.random()

        if question_all:
            random_number = random.choice([0.5, 0.9])

        if random_number < 0.3:
            question_all = True
            question = random.choice(QUESTION_ALL)
            if round == 0:
                conversation_list.append({"from": "human", "value": f"<image>\n{question}"})
            else:
                conversation_list.append({"from": "human", "value": f"{question}"})

            # translate the labels to categories
            label_each_pixel_diversity = [labels[label] for label in array_annotation]

            ####  index  ####
            label_each_pixel_diversity = np.reshape(label_each_pixel_diversity, (h_new, w_new))
            SEG = encode_mask(label_each_pixel_diversity)
            ####  index  ####

            answer = random.choice(ANSWER_ALL) + f"\n<seg>{SEG}</seg>"

            conversation_list.append({"from": "gpt", "value": answer})
        elif random_number < 0.65:
            question = random.choice(QUESTION_PARTIAL)

            # random select several categories in labels_in_image
            if random.random() < 0.5:
                partial_labels = random.sample(labels_in_image.tolist(), random.randint(1, min(len(labels_in_image), 3)))
            else:
                partial_labels = random.sample(labels_in_image.tolist(), random.randint(1, len(labels_in_image)))

            # translate the labels to categories
            partial_labels = [labels[partial_label] for partial_label in partial_labels]

            random.shuffle(partial_labels)

            question = question.replace("[class_name]", ", ".join(partial_labels))

            if round == 0:
                conversation_list.append({"from": "human", "value": f"<image>\n{question}"})
            else:
                conversation_list.append({"from": "human", "value": f"{question}"})

            label_each_pixel_partial = [label if label in partial_labels else "others" for label in label_each_pixel]

            ####  index  ####
            label_each_pixel_partial = np.reshape(label_each_pixel_partial, (h_new, w_new))
            SEG = encode_mask(label_each_pixel_partial)
            ####  index  ####

            answer = random.choice(ANSWER_PARTIAL).replace("[class_name]", ", ".join(partial_labels)) + f"\n<seg>{SEG}</seg>"

            conversation_list.append({"from": "gpt", "value": answer})
        else:
            question = random.choice(QUESTION_CONDITION)

            # random select several categories in labels_in_image
            if random.random() < 0.5:
                condition_labels = random.sample(labels_in_image.tolist(), random.randint(1, min(len(labels_in_image), 3)))
            else:
                condition_labels = random.sample(labels_in_image.tolist(), random.randint(1, len(labels_in_image)))

            # translate the labels to categories
            condition_labels = [labels[condition_label] for condition_label in condition_labels]

            # add others in condition_labels
            if random.random() < 0.5:
                condition_labels.append("others")

            # add some redundant labels form labels_not_in_image in condition_labels
            if random.random() < 0.5:
                redundant_labels = random.sample(labels_not_in_image, random.randint(1, 3))
                redundant_labels = [labels[redundant_label] for redundant_label in redundant_labels]
                condition_redundant_labels = condition_labels + redundant_labels
            else:
                condition_redundant_labels = condition_labels

            random.shuffle(condition_redundant_labels)

            question = question.replace("[class_name]", ", ".join(condition_redundant_labels))

            if round == 0:
                conversation_list.append({"from": "human", "value": f"<image>\n{question}"})
            else:
                conversation_list.append({"from": "human", "value": f"{question}"})

            label_each_pixel_condition = [label if label in condition_labels else "others" for label in label_each_pixel]

            ####  index  ####
            label_each_pixel_condition = np.reshape(label_each_pixel_condition, (h_new, w_new))
            SEG = encode_mask(label_each_pixel_condition)
            ####  index  ####

            answer = random.choice(ANSWER_CONDITION).replace("[class_name]", ", ".join(condition_redundant_labels)) + f"\n<seg>{SEG}</seg>"

            conversation_list.append({"from": "gpt", "value": answer})

        item["conversations"] = conversation_list

    Content.append(item)
    i += 1
    print(i)
    # if i > 5:
    #     break

with open("mapillary_two_round.json", "w") as f:
    json.dump(Content, f, indent=4)

