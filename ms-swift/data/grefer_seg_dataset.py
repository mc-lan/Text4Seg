from PIL import Image
import numpy as np
import torch
from pycocotools import mask
from data.grefer import G_REFER

class grefcocoValDataset(torch.utils.data.Dataset):
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        val_dataset,
    ):
        self.base_image_dir = base_image_dir
        splits = val_dataset.split("|")
        ds, splitBy, split = splits
        self.refer_api = G_REFER(self.base_image_dir, ds, splitBy)

        ref_ids = self.refer_api.getRefIds(split=split)
        images_ids = self.refer_api.getImgIds(ref_ids=ref_ids)  
        self.loaded_images = self.refer_api.loadImgs(image_ids=images_ids) 

    def __len__(self):
        return len(self.loaded_images)

    def __getitem__(self, idx):
        image_info = self.loaded_images[idx]
        image_path = self.base_image_dir + "images/coco_2014/train2014/" + image_info["file_name"]
        image = Image.open(image_path).convert('RGB')

        refs = self.refer_api.imgToRefs[image_info["id"]]

        masks = []
        questions = []
        for ref in refs:
            sentences = ref['sentences']
            anns = self.refer_api.refToAnn[ref['ref_id']]

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
                        pass
                    m = m + np.sum(mask.decode(rle), axis=2)
                m[m > 1] = 1
            m = torch.from_numpy(m.astype(np.uint8))
            for sentence in sentences:
                masks.append(m)
                questions.append(sentence['sent'])
        return (
            image,
            masks,
            questions,
            image_path
        )


if __name__ == "__main__":
    val_dataset = grefcocoValDataset(
        './datasets/refer_seg/',
        'grefcoco|unc|val'
    )

    print(len(val_dataset))
    data = val_dataset[3]
    image, masks, questions, image_path = data
    print(image_path)

    # visulize the image and mask
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    for i in range(len(masks)):
        plt.imshow(masks[i])
        plt.axis('off')
        plt.title(questions[i])
        plt.show()


