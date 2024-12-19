# -*- coding: utf-8 -*-
__author__ = "Chi Xie and Zhao Zhang"
__maintainer__ = "Chi Xie"
# this is the core of the d-cube toolkit
import os
import os.path as osp
import json
from collections import defaultdict

import numpy as np
from pycocotools import mask
import cv2
import matplotlib.pyplot as plt


from .data_util import *


class D3:
    def __init__(self, img_root, anno_root):
        self.image_dir = img_root
        self.anno_dir = anno_root
        self.load_data()

    def load_data(self):
        file_names = ["sentences.pkl", "annotations.pkl", "images.pkl", "groups.pkl"]
        self.data = {
            name.split(".")[0]: load_pkl(osp.join(self.anno_dir, name))
            for name in file_names
        }

    def get_sent_ids(self, anno_ids=[], img_ids=[], group_ids=[], sent_ids=[]):
        """get sentence ids for D-cube.

        Args:
            anno_ids (list, optional): annotation ids to get sentence ids. Defaults to [].
            img_ids (list, optional): image ids to get sentence ids. Defaults to [].
            group_ids (list, optional): group ids to get sentence ids. Defaults to [].
            sent_ids (list, optional): additional sentence ids you want to include. Defaults to [].

        Raises:
            Exception: anno_ids, img_ids and group_ids cannot be used together.

        Returns:
            list: sentence ids.
        """
        img_ids = img_ids if isinstance(img_ids, list) else [img_ids]
        anno_ids = anno_ids if isinstance(anno_ids, list) else [anno_ids]
        group_ids = group_ids if isinstance(group_ids, list) else [group_ids]
        sent_ids = sent_ids if isinstance(sent_ids, list) else [sent_ids]

        if not any([img_ids, anno_ids, group_ids, sent_ids]):
            return list(self.data["sentences"].keys())

        if (
            (anno_ids and img_ids)
            or (anno_ids and group_ids)
            or (img_ids and group_ids)
        ):
            raise Exception("anno_ids, img_ids, group_ids can only be used alone")

        out_ids_set = set()
        if img_ids:
            for img_id in img_ids:
                imganno_ids = self.data["images"][img_id]["anno_id"]
                for ianno_id in imganno_ids:
                    out_ids_set |= set(self.data["annotations"][ianno_id]["sent_id"])

        if group_ids:
            for group_id in group_ids:
                out_ids_set |= set(self.data["groups"][group_id]["inner_sent_id"])

        if anno_ids:
            for ianno_id in anno_ids:
                out_ids_set |= set(self.data["annotations"][ianno_id]["sent_id"])

        if sent_ids:
            out_ids_set &= set(sent_ids)

        return list(out_ids_set)

    def get_anno_ids(self, anno_ids=[], img_ids=[], group_ids=[], sent_ids=[]):
        """get annotation ids for D-cube.

        Args:
            anno_ids (list, optional): additional annotation ids you want to include. Defaults to [].
            img_ids (list, optional): image ids to get annotation ids. Defaults to [].
            group_ids (list, optional): group ids to get annotation ids. Defaults to [].
            sent_ids (list, optional): sentence ids to get annotation ids. Defaults to [].

        Raises:
            Exception: img_ids and group_ids cannot be used together.

        Returns:
            list: annotation ids.
        """
        img_ids = img_ids if isinstance(img_ids, list) else [img_ids]
        anno_ids = anno_ids if isinstance(anno_ids, list) else [anno_ids]
        group_ids = group_ids if isinstance(group_ids, list) else [group_ids]
        sent_ids = sent_ids if isinstance(sent_ids, list) else [sent_ids]

        if not any([img_ids, anno_ids, group_ids, sent_ids]):
            return list(self.data["annotations"].keys())

        if img_ids and group_ids:
            raise Exception("img_ids, group_ids can only be used alone")

        out_ids_set = set()
        if img_ids:
            for img_id in img_ids:
                out_ids_set |= set(self.data["images"][img_id]["anno_id"])

        if group_ids:
            for group_id in group_ids:
                for groupimg_id in self.data["groups"][group_id]["img_id"]:
                    out_ids_set |= set(self.data["images"][groupimg_id]["anno_id"])

        if sent_ids and img_ids:
            for sent_id in sent_ids:
                out_ids_set &= set(self.data["sentences"][sent_id]["anno_id"])
        else:
            for sent_id in sent_ids:
                out_ids_set |= set(self.data["sentences"][sent_id]["anno_id"])

        if anno_ids:
            out_ids_set &= set(anno_ids)

        return list(out_ids_set)

    def get_img_ids(self, anno_ids=[], img_ids=[], group_ids=[], sent_ids=[]):
        """get image ids for D-cube.

        Args:
            anno_ids (list, optional): annotation ids to get image ids. Defaults to [].
            img_ids (list, optional): additional image ids you want to include. Defaults to [].
            group_ids (list, optional): group ids to get image ids. Defaults to [].
            sent_ids (list, optional): sentence ids to get image ids. Defaults to [].

        Raises:
            Exception: anno_ids and img_ids cannot be used together.
            Exception: anno_ids and group_ids cannot be used together.

        Returns:
            list: image ids.
        """
        img_ids = img_ids if isinstance(img_ids, list) else [img_ids]
        anno_ids = anno_ids if isinstance(anno_ids, list) else [anno_ids]
        group_ids = group_ids if isinstance(group_ids, list) else [group_ids]
        sent_ids = sent_ids if isinstance(sent_ids, list) else [sent_ids]

        if not any([img_ids, anno_ids, group_ids, sent_ids]):
            return list(self.data["images"].keys())

        if anno_ids and img_ids:
            raise Exception("anno_ids and img_ids can only be used alone")
        if anno_ids and group_ids:
            raise Exception("anno_ids and group_ids can only be used alone")

        out_ids_set = set()
        if anno_ids:
            for ianno_id in anno_ids:
                out_ids_set.add(self.data["annotations"][ianno_id]["image_id"])

        if group_ids:
            for group_id in group_ids:
                out_ids_set |= set(self.data["groups"][group_id]["img_id"])

        if sent_ids:
            for sent_id in sent_ids:
                for sentanno_id in self.data["sentences"][sent_id]["anno_id"]:
                    out_ids_set.add(self.data["annotations"][sentanno_id]["image_id"])

        if img_ids:
            out_ids_set &= set(img_ids)

        return list(out_ids_set)

    def get_group_ids(self, anno_ids=[], img_ids=[], group_ids=[], sent_ids=[]):
        """get group ids for D-cube.

        Args:
            anno_ids (list, optional): annotation ids to get group ids. Defaults to [].
            img_ids (list, optional): image ids to get group ids. Defaults to [].
            group_ids (list, optional): additional group_ids you want to include. Defaults to [].
            sent_ids (list, optional): sentence ids to get group ids. Defaults to [].

        Raises:
            Exception: anno_ids, img_ids and sent_ids cannot be used together.

        Returns:
            list: group ids.
        """
        img_ids = img_ids if isinstance(img_ids, list) else [img_ids]
        anno_ids = anno_ids if isinstance(anno_ids, list) else [anno_ids]
        group_ids = group_ids if isinstance(group_ids, list) else [group_ids]
        sent_ids = sent_ids if isinstance(sent_ids, list) else [sent_ids]

        if not any([img_ids, anno_ids, group_ids, sent_ids]):
            return list(self.data["groups"].keys())

        if anno_ids and img_ids:
            raise Exception("anno_ids and img_ids can only be used alone")
        if anno_ids and sent_ids:
            raise Exception("anno_ids and sent_ids can only be used alone")
        if img_ids and sent_ids:
            raise Exception("img_ids and sent_ids can only be used alone")

        out_ids_set = set()
        if img_ids:
            for img_id in img_ids:
                out_ids_set.add(self.data["images"][img_id]["group_id"])

        if anno_ids:
            for anno_id in anno_ids:
                out_ids_set.add(self.data["annotations"][anno_id]["group_id"])

        if sent_ids:
            for sent_id in sent_ids:
                out_ids_set |= set(self.data["sentences"][sent_id]["group_id"])

        if group_ids:
            out_ids_set &= set(group_ids)

        return list(out_ids_set)

    def load_sents(self, sent_ids=None):
        """load sentence info.

        Args:
            sent_ids (list, int, optional): sentence ids. Defaults to None.

        Returns:
            list: a list of sentence info.
        """
        if sent_ids is not None and not isinstance(sent_ids, list):
            sent_ids = [sent_ids]
        if isinstance(sent_ids, list):
            return [self.data["sentences"][sent_id] for sent_id in sent_ids]
        else:
            return list(self.data["sentences"].values())

    def load_annos(self, anno_ids=None):
        """load annotation info.

        Args:
            anno_ids (list, int, optional): annotation ids. Defaults to None.

        Returns:
            list: a list of annotation info.
        """
        if anno_ids is not None and not isinstance(anno_ids, list):
            anno_ids = [anno_ids]
        if isinstance(anno_ids, list):
            return [self.data["annotations"][anno_id] for anno_id in anno_ids]
        else:
            return list(self.data["annotations"].values())

    def load_imgs(self, img_ids=None):
        """load image info.

        Args:
            img_ids (list, int, optional): image ids. Defaults to None.

        Returns:
            list: a list of image info.
        """
        if img_ids is not None and not isinstance(img_ids, list):
            img_ids = [img_ids]
        if isinstance(img_ids, list):
            return [self.data["images"][img_ids] for img_ids in img_ids]
        else:
            return list(self.data["images"].values())

    def load_groups(self, group_ids=None):
        """load group info.

        Args:
            group_ids (list, int, optional): group ids. Defaults to None.

        Returns:
            list: a list of group info.
        """
        if group_ids is not None and not isinstance(group_ids, list):
            group_ids = [group_ids]
        if isinstance(group_ids, list):
            return [self.data["groups"][group_ids] for group_ids in group_ids]
        else:
            return list(self.data["groups"].values())

    def get_mask(self, anno):
        rle = anno[0]["segmentation"]
        m = mask.decode(rle)
        m = np.sum(
            m, axis=2
        )  # sometimes there are multiple binary map (corresponding to multiple segs)
        m = m.astype(np.uint8)  # convert to np.uint8
        # compute area
        area = sum(mask.area(rle))  # should be close to ann['area']
        return {"mask": m, "area": area}

    def show_mask(self, anno):
        M = self.get_mask(anno)
        msk = M["mask"]
        ax = plt.gca()
        ax.imshow(msk)

    def show_image_seg(
        self,
        img_ids=[],
        save_dir=None,
        show_sent=False,
        on_image=False,
        checkerboard_bg=False,
        is_instance=True,
    ):
        if is_instance and checkerboard_bg:
            raise ValueError(
                "Cannot apply both is_instance and checkboard_bg at the same time."
            )
        img_infos = self.load_imgs(img_ids=img_ids)
        for img_idx, img_info in enumerate(img_infos):
            img = cv2.imread(osp.join(self.image_dir, img_info["file_name"]))
            anno_infos = self.load_annos(img_info["anno_id"])

            bm_canvas = defaultdict(list)
            merge_canvas = defaultdict(list)
            for anno_info in anno_infos:
                for sent_id in anno_info["sent_id"]:
                    bm_canvas[sent_id].append(anno_info["segmentation"])

            for sent_id, bm_list in bm_canvas.items():
                merge_canvas[sent_id] = merge_rle(
                    bm_list, is_instance=is_instance, on_image=on_image
                )

            cv2.imwrite(osp.join(save_dir, f"{img_info['id']}.png"), img)
            for sent_id, merge_mask in merge_canvas.items():
                if checkerboard_bg:
                    merge_mask = add_checkerboard_bg(img, merge_mask)
                elif on_image:
                    merge_mask = visualize_mask_on_image(img, merge_mask, add_edge=True)
                if show_sent:
                    sent_en = self.load_sents(sent_ids=sent_id)[0]["raw_sent"]
                    merge_mask = paste_text(merge_mask, sent_en)
                cv2.imwrite(
                    osp.join(save_dir, f"{img_info['id']}_{sent_id}.png"), merge_mask
                )

        return merge_canvas

    def show_group_seg(
        self,
        group_ids,
        save_root,
        show_sent=True,
        is_instance=True,
        on_image=False,
        checkerboard_bg=False,
    ):
        group_infos = self.load_groups(group_ids=group_ids)
        for group_info in group_infos:
            save_dir = osp.join(save_root, group_info["group_name"])
            os.makedirs(save_dir, exist_ok=True)
            self.show_image_seg(
                img_ids=group_info["img_id"],
                save_dir=save_dir,
                show_sent=show_sent,
                is_instance=is_instance,
                on_image=on_image,
                checkerboard_bg=checkerboard_bg,
            )

    def show_image_seg_bbox(
        self,
        img_ids=[],
        save_dir=None,
        show_sent=False,
        on_image=False,
        checkerboard_bg=False,
        is_instance=True,
    ):
        if is_instance and checkerboard_bg:
            raise ValueError(
                "Cannot apply both is_instance and checkboard_bg at the same time."
            )
        img_infos = self.load_imgs(img_ids=img_ids)
        for img_idx, img_info in enumerate(img_infos):
            img = cv2.imread(osp.join(self.image_dir, img_info["file_name"]))
            anno_infos = self.load_annos(img_info["anno_id"])

            bm_canvas = defaultdict(list)
            merge_canvas = defaultdict(list)
            sent_boxes = defaultdict(list)
            for anno_info in anno_infos:
                for sent_id in anno_info["sent_id"]:
                    bm_canvas[sent_id].append(anno_info["segmentation"])
                    sent_boxes[sent_id].append(anno_info["bbox"][0].tolist())

            for sent_id, bm_list in bm_canvas.items():
                merge_canvas[sent_id] = merge_rle(
                    bm_list, is_instance=is_instance, on_image=on_image
                )

            cv2.imwrite(osp.join(save_dir, f"{img_info['id']}.png"), img)
            for sent_id, merge_mask in merge_canvas.items():
                # vis mask
                if checkerboard_bg:
                    merge_mask = add_checkerboard_bg(img, merge_mask)
                elif on_image:
                    merge_mask = visualize_mask_on_image(img, merge_mask, add_edge=True)
                # vis box
                bboxes = sent_boxes[sent_id]
                merge_mask = visualize_bbox_on_image(merge_mask, bboxes)
                # vis sent
                if show_sent:
                    sent_en = self.load_sents(sent_ids=sent_id)[0]["raw_sent"]
                    merge_mask = paste_text(merge_mask, sent_en)
                cv2.imwrite(
                    osp.join(save_dir, f"{img_info['id']}_{sent_id}.png"), merge_mask
                )

        return merge_canvas

    def show_group_seg_bbox(
        self,
        group_ids,
        save_root,
        show_sent=True,
        is_instance=True,
        on_image=False,
        checkerboard_bg=False,
    ):
        group_infos = self.load_groups(group_ids=group_ids)
        for group_info in group_infos:
            save_dir = osp.join(save_root, group_info["group_name"])
            os.makedirs(save_dir, exist_ok=True)
            self.show_image_seg_bbox(
                img_ids=group_info["img_id"],
                save_dir=save_dir,
                show_sent=show_sent,
                is_instance=is_instance,
                on_image=on_image,
                checkerboard_bg=checkerboard_bg,
            )

    def show_image_bbox(self, img_ids=[], save_dir=None, show_sent=False):
        img_infos = self.load_imgs(img_ids=img_ids)
        for img_idx, img_info in enumerate(img_infos):
            img = cv2.imread(osp.join(self.image_dir, img_info["file_name"]))
            anno_infos = self.load_annos(img_info["anno_id"])

            sent_boxes = defaultdict(list)
            for anno_info in anno_infos:
                for sent_id in anno_info["sent_id"]:
                    sent_boxes[sent_id].append(anno_info["bbox"][0].tolist())

            cv2.imwrite(osp.join(save_dir, f"{img_info['id']}.png"), img)
            for sent_id, bboxes in sent_boxes.items():
                merge_img = visualize_bbox_on_image(img, bboxes)
                if show_sent:
                    sent_en = self.load_sents(sent_ids=sent_id)[0]["raw_sent"]
                    merge_img = paste_text(merge_img, sent_en)
                cv2.imwrite(
                    osp.join(save_dir, f"{img_info['id']}_{sent_id}.png"), merge_img
                )

    def show_group_bbox(self, group_ids, save_root, show_sent=True):
        group_infos = self.load_groups(group_ids=group_ids)
        for group_info in group_infos:
            save_dir = osp.join(save_root, group_info["group_name"])
            os.makedirs(save_dir, exist_ok=True)
            self.show_image_bbox(
                img_ids=group_info["img_id"], save_dir=save_dir, show_sent=show_sent
            )

    def stat_description(self, with_rev=False, inter_group=False):
        """calculate and print dataset statistics.

        Args:
            with_rev (bool, optional): consider absence descriptions or not. Defaults to False.
            inter_group (bool, optional): calculate under intra- or inter-group settings. Defaults to False.
        """
        stat_dict = {}
        # Number of sents
        sent_ids = list(self.data["sentences"].keys())
        if not with_rev:
            sent_ids = [sent_id for sent_id in sent_ids if not self.is_revsent(sent_id)]
        stat_dict["nsent"] = len(sent_ids)
        # Number of annos / instance  # TODO: rm rev
        stat_dict["nanno"] = len(self.data["annotations"].keys())
        # Number of images
        stat_dict["nimg"] = len(self.data["images"].keys())
        # Number of groups
        stat_dict["ngroup"] = len(self.data["groups"].keys())

        # Number of img-sent pair
        num_img_sent = 0
        for img_id in self.data["images"].keys():
            anno_ids = self.get_anno_ids(img_ids=img_id)
            anno_infos = self.load_annos(anno_ids=anno_ids)
            cur_sent_set = set()
            group_sent_ids = set(
                self.load_groups(self.get_group_ids(img_ids=img_id))[0]["inner_sent_id"]
            )
            for anno_info in anno_infos:
                cur_sent_set |= set(
                    [i for i in anno_info["sent_id"] if i in group_sent_ids]
                )
            if not with_rev:
                cur_sent_set = [
                    sent_id for sent_id in cur_sent_set if not self.is_revsent(sent_id)
                ]
            num_img_sent += len(cur_sent_set)
        stat_dict["num_img_sent"] = num_img_sent

        # Number of absence img-sent pair
        num_anti_img_sent = 0
        for img_id in self.data["images"].keys():
            anno_ids = self.get_anno_ids(img_ids=img_id)
            anno_infos = self.load_annos(anno_ids=anno_ids)
            cur_sent_set = set()
            group_sent_ids = set(
                self.load_groups(self.get_group_ids(img_ids=img_id))[0]["inner_sent_id"]
            )
            for anno_info in anno_infos:
                cur_sent_set |= set(
                    [i for i in anno_info["sent_id"] if i in group_sent_ids]
                )
            assert group_sent_ids.issuperset(
                cur_sent_set
            ), f"{group_sent_ids}, {cur_sent_set}"
            cur_anti_sent_set = group_sent_ids - cur_sent_set
            if not with_rev:
                cur_anti_sent_set = [
                    sent_id
                    for sent_id in cur_anti_sent_set
                    if not self.is_revsent(sent_id)
                ]
            num_anti_img_sent += len(cur_anti_sent_set)
        stat_dict["num_anti_img_sent"] = num_anti_img_sent

        # Number of anno-sent pair
        num_anno_sent = 0
        anno_infos = self.load_annos()
        for anno_info in anno_infos:
            if inter_group:
                anno_sent_ids = [i for i in anno_info["sent_id"]]
            else:
                group_sent_ids = set(
                    self.load_groups(anno_info["group_id"])[0]["inner_sent_id"]
                )
                anno_sent_ids = [i for i in anno_info["sent_id"] if i in group_sent_ids]
            if not with_rev:
                anno_sent_ids = [
                    sent_id for sent_id in anno_sent_ids if not self.is_revsent(sent_id)
                ]
            num_anno_sent += len(anno_sent_ids)

        stat_dict["num_anno_sent"] = num_anno_sent

        # Number of anti anno-sent pair
        num_anti_anno_sent = 0
        anno_infos = self.load_annos()
        for anno_info in anno_infos:
            if inter_group:
                all_sent_ids = set(self.get_sent_ids())
                anno_sent_ids = anno_info["sent_id"]

                anti_sent_ids = [
                    sent_id for sent_id in all_sent_ids if sent_id not in anno_sent_ids
                ]
            else:
                group_sent_ids = set(
                    self.load_groups(anno_info["group_id"])[0]["inner_sent_id"]
                )
                anno_sent_ids = [i for i in anno_info["sent_id"] if i in group_sent_ids]

                anti_sent_ids = [
                    sent_id
                    for sent_id in group_sent_ids
                    if sent_id not in anno_sent_ids
                ]

            if not with_rev:
                anti_sent_ids = [
                    sent_id for sent_id in anti_sent_ids if not self.is_revsent(sent_id)
                ]
            num_anti_anno_sent += len(anti_sent_ids)

        stat_dict["num_anti_anno_sent"] = num_anti_anno_sent

        # Len of sentence
        totle_len = 0
        for sent_info in self.load_sents(sent_ids):
            totle_len += len(sent_info["raw_sent"].split())

        stat_dict["avg_sent_len"] = totle_len / stat_dict["nsent"]

        print(stat_dict)

    def is_revsent(self, sent_id):
        sent_info = self.load_sents(sent_ids=sent_id)
        return sent_info[0]["is_negative"]

    def data2coca(self, out_root, with_rev=False):
        group_infos = self.load_groups()
        for group_info in group_infos:
            sent_ids = group_info["inner_sent_id"]
            if not with_rev:
                sent_ids = [
                    sent_id for sent_id in sent_ids if not self.is_revsent(sent_id)
                ]
            sent_infos = self.load_sents(sent_ids)
            for sent_info in sent_infos:
                sent = sent_info["raw_sent"]
                img_infos = self.load_imgs(group_info["img_id"])
                for img_info in img_infos:
                    src_img_path = osp.join(self.image_dir, img_info["file_name"])
                    raw_name = img_info["file_name"].split("/")[-1]
                    out_img_dir = osp.join(out_root, "images", sent)
                    os.makedirs(out_img_dir, exist_ok=True)
                    out_img_path = osp.join(out_img_dir, raw_name)
                    copy_file(src_img_path, out_img_path)

                    out_mask_dir = osp.join(out_root, "masks", sent)
                    os.makedirs(out_mask_dir, exist_ok=True)
                    out_mask_path = osp.join(
                        out_mask_dir, raw_name.replace(".jpg", ".png")
                    )

                    cur_anno_ids = self.get_anno_ids(
                        img_ids=img_info["id"], sent_ids=sent_info["id"]
                    )
                    anno_infos = self.load_annos(cur_anno_ids)
                    rle_list = [anno_info["segmentation"] for anno_info in anno_infos]
                    bmask = merge2bin(rle_list, img_info["height"], img_info["width"])
                    cv2.imwrite(out_mask_path, bmask)

    def convert2coco(self, out_root, anti_mode=False, is_group_separated=True):
        """
        Convert the annotation format of D^3 dataset to COCO.
        1. The sent_id can be viewed as category_id in COCO.
        2. If `is_group_separated` is True, `outer_sent_id` does not need to be considered.
        3. if `with_rev` is False,  sents that meet `is_revsent` will be ignore.
        """
        os.makedirs(out_root, exist_ok=True)
        coco_dict = {
            "images": [],
            "categories": [],
            "annotations": [],
        }

        sent_ids = self.get_sent_ids()
        if anti_mode == 1:
            sent_ids = [sent_id for sent_id in sent_ids if not self.is_revsent(sent_id)]
        elif anti_mode == 2:
            sent_ids = [sent_id for sent_id in sent_ids if self.is_revsent(sent_id)]
        elif anti_mode == 0:
            pass
        else:
            raise Exception("Unimplemented anti_mode.")

        sent_infos = self.load_sents(sent_ids)
        for isent_info in sent_infos:
            coco_dict["categories"].append(
                {
                    "id": isent_info["id"],
                    "name": isent_info["raw_sent"],
                }
            )

        item_id = 0
        img_infos = self.load_imgs()
        for iimg_info in img_infos:
            coco_dict["images"].append(
                {
                    "id": iimg_info["id"],
                    "file_name": iimg_info["file_name"],
                    "height": iimg_info["height"],
                    "width": iimg_info["width"],
                }
            )

            anno_ids = self.get_anno_ids(img_ids=iimg_info["id"])
            anno_infos = self.load_annos(anno_ids)

            for ianno_info in anno_infos:
                if is_group_separated:
                    inner_group_sent_ids = [
                        isent_id
                        for isent_id in ianno_info["sent_id"]
                        if isent_id
                        in self.load_groups(ianno_info["group_id"])[0]["inner_sent_id"]
                    ]
                    cur_sent_ids = inner_group_sent_ids
                else:
                    cur_sent_ids = ianno_info["sent_id"]

                for isent_id in cur_sent_ids:
                    if isent_id not in sent_ids:
                        continue

                    seg = ianno_info["segmentation"][0].copy()
                    if isinstance(seg, dict):  # RLE
                        counts = seg["counts"]
                        if not isinstance(counts, str):
                            # make it json-serializable
                            seg["counts"] = counts.decode("ascii")

                    coco_dict["annotations"].append(
                        {
                            "id": item_id,
                            "image_id": iimg_info["id"],
                            "category_id": isent_id,
                            "segmentation": seg,
                            "area": int(ianno_info["area"][0]),
                            "bbox": [
                                int(cord) for cord in ianno_info["bbox"][0].tolist()
                            ],
                            "iscrowd": 0,  # TODO: ianno_info["iscrowd"]
                        }
                    )
                    item_id += 1

        with open(osp.join(out_root, "coco_annotations.json"), "w") as f:
            json.dump(coco_dict, f, indent=4)

    def sent_analyse(self, save_dir, with_rev=False):
        """analyze word info in D-cube and generate word length histograms, word clouds, etc.

        Args:
            save_dir (str): path to save the visualized results.
            with_rev (bool, optional): consider absence descriptions or not. Defaults to False.
        """
        sent_ids = self.get_sent_ids()
        if not with_rev:
            sent_ids = [sent_id for sent_id in sent_ids if not self.is_revsent(sent_id)]

        sent_lens, sent_raws = [], []
        sent_infos = self.load_sents(sent_ids)
        for isent_info in sent_infos:
            sent_raws.append(isent_info["raw_sent"])
            sent_lens.append(len(isent_info["raw_sent"].split()))

        os.makedirs(save_dir, exist_ok=True)
        # plot_hist(
        #     sent_lens,
        #     bins=max(sent_lens) - min(sent_lens) + 1,
        #     save_path=osp.join(save_dir, "words_hist.pdf"),
        #     x="Lengths of descriptions",
        # )
        # generate_wordclouds(sent_raws, osp.join(save_dir, "word_clouds"))

    def group_analysis(self, save_dir, with_rev=False):
        group_infos = self.load_groups()
        scene_tree = defaultdict(dict)

        for group_info in group_infos:
            scene_tree[group_info["scene"]][group_info["group_name"]] = {"nimg": 0.1}

        # vis_group_tree(scene_tree, osp.join(save_dir, 'scene_tree.png'))  # the visualized result is ugly

    def bbox_num_analyze(self):
        n_cat = len(self.data["sentences"].keys())
        all_img_ids = self.data["images"].keys()
        n_img = len(all_img_ids)
        cat_obj_count = np.zeros((n_cat, n_img), dtype=int)
        for img_id in all_img_ids:
            # img_cat_ids = self.get_sent_ids(img_ids=img_id)
            anno_ids = self.get_anno_ids(img_ids=img_id)
            anno_infos = self.load_annos(anno_ids=anno_ids)
            for anno in anno_infos:
                for sid in anno["sent_id"]:
                    cat_obj_count[sid - 1, img_id] += 1
        return cat_obj_count
