import os
import numpy as np
from PIL import Image
import random
from question_answer_list import QUESTION_ALL, QUESTION_PARTIAL, QUESTION_CONDITION, ANSWER_ALL, ANSWER_PARTIAL, ANSWER_CONDITION
from utils import encode_mask, random_crop

h_new = w_new = 32
images_path = "./playground/data/coco/train2017"
annotations_path = "./playground/data/coco_stuff/annotations"

# read cls_coco_stuff.txt to get the labels
labels = {}
with open("./playground/data/create_json/cls_coco_stuff.txt") as f:
    for idx, line in enumerate(f):
        labels[idx] = line.strip()

Content = []
i = 0
for _ in range(10):
    for image_file in os.listdir(images_path):
        item = {}
        item["id"] = image_file.split(".")[0]
        item["image"] = "coco/train2017/" + image_file

        annotation = Image.open(os.path.join(annotations_path, image_file.replace(".jpg", "_labelTrainIds.png")))

        annotation, crop_params = random_crop(annotation)

        item["crop_size"] = crop_params
        item['height'] = h_new
        item['width'] = w_new

        annotation = annotation.resize((w_new, h_new), Image.NEAREST)  
        array_annotation = np.array(annotation) 
        array_annotation = array_annotation.flatten()

        # translate the labels to categories: exclude 255
        label_each_pixel = [labels[label].split(", ")[0] if label in labels else "others" for label in array_annotation]

        labels_in_image = np.unique(array_annotation)

        if len(labels_in_image) == 1:
            continue

        # remove 255 from labels_in_image
        labels_in_image = labels_in_image[labels_in_image != 255]
        labels_not_in_image = [label for label in range(171) if label not in labels_in_image]

        # conversations
        conversation_list = []

        question_all = False

        for round in range(2):
            # select one quesetion from questions_all or questions_partial
            random_number = random.random()

            if question_all:
                random_number = random.choice([0.4, 0.9])

            if random_number < 0.1:
                question_all = True
                question = random.choice(QUESTION_ALL)
                if round == 0:
                    conversation_list.append({"from": "human", "value": f"<image>\n{question}"})
                else:
                    conversation_list.append({"from": "human", "value": f"{question}"})

                # translate the labels to categories
                label_each_pixel_diversity = []
                label_diversity = labels.copy()
                for label in labels_in_image:
                    if random.random() < 0.2:
                        label_diversity[label] = labels[label].split(", ")[0]
                    else:
                        try:
                            label_diversity[label] = random.choice(labels[label].split(", ")[1:])
                        except:
                            label_diversity[label] = labels[label].split(", ")[0]

                # if label == 255, then label_diversity[label] = "others"
                label_each_pixel_diversity = [label_diversity[label] if label in label_diversity else "others" for label in array_annotation]

                ####  index  ####
                label_each_pixel_diversity = np.reshape(label_each_pixel_diversity, (h_new, w_new))
                SEG = encode_mask(label_each_pixel_diversity)
                ####  index  ####

                answer = random.choice(ANSWER_ALL) + f"\n<seg>{SEG}</seg>"

                conversation_list.append({"from": "gpt", "value": answer})
            elif random_number < 0.4:
                question = random.choice(QUESTION_PARTIAL)

                partial_labels = random.sample(labels_in_image.tolist(), random.randint(1, len(labels_in_image)))

                # translate the labels to categories
                partial_labels = [labels[partial_label].split(', ')[0] for partial_label in partial_labels]

                random.shuffle(partial_labels)

                question = question.replace("[class_name]", "|".join(partial_labels))

                if round == 0:
                    conversation_list.append({"from": "human", "value": f"<image>\n{question}"})
                else:
                    conversation_list.append({"from": "human", "value": f"{question}"})

                label_each_pixel_partial = [label if label in partial_labels else "others" for label in label_each_pixel]

                ####  index  ####
                label_each_pixel_partial = np.reshape(label_each_pixel_partial, (h_new, w_new))
                SEG = encode_mask(label_each_pixel_partial)
                ####  index  ####

                answer = random.choice(ANSWER_PARTIAL).replace("[class_name]", "|".join(partial_labels)) + f"\n<seg>{SEG}</seg>"

                conversation_list.append({"from": "gpt", "value": answer})
            else:
                question = random.choice(QUESTION_CONDITION)

                condition_labels = random.sample(labels_in_image.tolist(), random.randint(1, len(labels_in_image)))

                # translate the labels to categories
                condition_labels = [labels[condition_label].split(', ')[0] for condition_label in condition_labels]

                # add others in condition_labels
                if random.random() < 0.5:
                    condition_labels.append("others")

                # add some redundant labels form labels_not_in_image in condition_labels
                if random.random() < 0.9:
                    redundant_labels = random.sample(labels_not_in_image, random.randint(1, len(labels_not_in_image)))
                    redundant_labels = [labels[redundant_label].split(', ')[0] for redundant_label in redundant_labels]
                    condition_redundant_labels = condition_labels + redundant_labels
                else:
                    condition_redundant_labels = condition_labels

                random.shuffle(condition_redundant_labels)

                question = question.replace("[class_name]", "|".join(condition_redundant_labels))

                if round == 0:
                    conversation_list.append({"from": "human", "value": f"<image>\n{question}"})
                else:
                    conversation_list.append({"from": "human", "value": f"{question}"})

                label_each_pixel_condition = [label if label in condition_labels else "others" for label in label_each_pixel]

                ####  index  ####
                label_each_pixel_condition = np.reshape(label_each_pixel_condition, (h_new, w_new))
                SEG = encode_mask(label_each_pixel_condition)
                ####  index  ####

                answer = random.choice(ANSWER_CONDITION).replace("[class_name]", "|".join(condition_redundant_labels)) + f"\n<seg>{SEG}</seg>"

                conversation_list.append({"from": "gpt", "value": answer})

            item["conversations"] = conversation_list

        Content.append(item)
        i += 1
        print(i)

import json
with open("./playground/data/json_files/cocostuff_32_two_round_10.json", "w") as f:
    json.dump(Content, f, indent=4)