import os
import random
import glob
from PIL import Image

import cv2
import numpy as np
import torch
from pycocotools import mask

# from .create_json.grefer import G_REFER
from llava.eval.refer import REFER
# from .utils import ANSWER_LIST, SHORT_QUESTION_LIST


class ValDataset(torch.utils.data.Dataset):
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        val_dataset,
    ):
        self.base_image_dir = base_image_dir
        splits = val_dataset.split("|")
        if len(splits) == 2:
            ds, split = splits
            images = glob.glob(
                os.path.join(self.base_image_dir, "reason_seg", ds, split, "*.jpg")
            )
            self.images = images
            self.data_type = "reason_seg"
        elif len(splits) == 3:
            ds, splitBy, split = splits
            refer_api = REFER(self.base_image_dir, ds, splitBy)
            ref_ids_val = refer_api.getRefIds(split=split)
            images_ids_val = refer_api.getImgIds(ref_ids=ref_ids_val)
            refs_val = refer_api.loadRefs(ref_ids=ref_ids_val)
            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_val)
            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(
                        base_image_dir, "images/saiapr_tc-12", item["file_name"]
                    )
                elif ds in ["refcoco", "refcoco+", "refcocog", "grefcoco"]:
                    item["file_name"] = os.path.join(
                        base_image_dir,
                        "images/coco_2014/train2014",
                        item["file_name"],
                    )
                refer_seg_ds["images"].append(item)
            refer_seg_ds["annotations"] = refer_api.Anns  # anns_val

            img2refs = {}
            for ref in refs_val:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [
                    ref,
                ]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_ds = refer_seg_ds
            self.data_type = "refer_seg"

        # self.ds = ds

    def __len__(self):
        if self.data_type == "refer_seg":
            return len(self.refer_seg_ds["images"])
        else:
            return len(self.images)

    def __getitem__(self, idx):
        if self.data_type == "refer_seg":
            refer_seg_ds = self.refer_seg_ds
            images = refer_seg_ds["images"]
            annotations = refer_seg_ds["annotations"]
            img2refs = refer_seg_ds["img2refs"]

            image_info = images[idx]
            image_path = image_info["file_name"]
            image_id = image_info["id"]

            refs = img2refs[image_id]
            if len(refs) == 0:
                raise ValueError("image {} has no refs".format(image_id))

            sents = []
            ann_ids = []
            for ref in refs:
                for sent in ref["sentences"]:
                    sents.append(sent["sent"].strip().lower())
                    ann_ids.append(ref["ann_id"])

            sampled_sents = sents
            sampled_ann_ids = ann_ids
            image = Image.open(image_path).convert('RGB')
            is_sentence = False
        # else:
        #     image_path = self.images[idx]
        #     image = cv2.imread(image_path)
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #     json_path = image_path.replace(".jpg", ".json")
        #     mask_json, sampled_sents, is_sentence = get_mask_from_json(json_path, image)
        #     sampled_sents = [sampled_sents[0]]

        i = 0
        questions = []
        while i < len(sampled_sents):
            text = sampled_sents[i].strip()
            questions.append(text)
            i += 1

        if self.data_type == "refer_seg":
            masks = []
            for i, ann_id in enumerate(sampled_ann_ids):
                ann = annotations[ann_id]
                if len(ann["segmentation"]) == 0 and sampled_sents[i] != "":
                    m = np.zeros((image_info["height"], image_info["width"], 1))
                else:
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
                m = np.sum(
                    m, axis=2
                )  # sometimes there are multiple binary map (corresponding to multiple segs)
                m = m.astype(np.uint8)  # convert to np.uint8
                masks.append(m)
        # else:
        #     masks = [mask_json]

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)

        return (
            image,
            masks,
            questions,
            image_path
        )


if __name__ == "__main__":
    val_dataset = ValDataset(
        '/data/mclan/LLaVa/playground/data/refer_seg/',
        'refcocog|umd|test'
    )
    # refcoco | unc | testA, testB, val,
    # refcoco+ | unc | testA, testB, val,
    # refcocog | umd | test, val
    # val_dataloader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=1,
    #     shuffle=False,
    # )

    print(len(val_dataset))
    for data in val_dataset:
        image, masks, questions, image_path = data
        # print(image.shape, masks.shape, questions, image_path)
        print(image_path)


        # visulize the image and mask
        # import matplotlib.pyplot as plt
        # plt.imshow(image)
        # plt.axis('off')
        # plt.show()
        #
        # for i in range(masks.shape[0]):
        #     plt.imshow(masks[i])
        #     plt.axis('off')
        #     plt.title(questions[i])
        #     plt.show()

        break

        # plt.imshow(masks[0])
        # plt.axis('off')
        # plt.show()
        # print(questions[0])

