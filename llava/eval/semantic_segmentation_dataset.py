import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Custom Dataset class
class ADE20KDataset(Dataset):
    def __init__(self, images_path, annotations_path, file_list_path=None):
        self.images_path = images_path
        self.annotations_path = annotations_path
        self.images_list = sorted(os.listdir(images_path))
        self.annotations_list = sorted(os.listdir(annotations_path))
        self.file_list = [img.split('.')[0] for img in self.images_list]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Get the image ID from the validation list
        image_id = self.file_list[idx]

        # Load the image
        image_path = os.path.join(self.images_path, f"{image_id}.jpg")
        image = Image.open(image_path).convert("RGB")

        # Load the annotation (mask)
        annotation_path = os.path.join(self.annotations_path, f"{image_id}.png")
        annotation = Image.open(annotation_path)

        return image, annotation, image_id


class PAS20Dataset(Dataset):
    def __init__(self, images_path, annotations_path, file_list_path):
        self.images_path = images_path
        self.annotations_path = annotations_path

        # Load the validation file list
        with open(file_list_path, 'r') as file:
            self.file_list = file.read().splitlines()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Get the image ID from the validation list
        image_id = self.file_list[idx]

        # Load the image
        image_path = os.path.join(self.images_path, f"{image_id}.jpg")
        image = Image.open(image_path).convert("RGB")

        # Load the annotation (segmentation mask)
        annotation_path = os.path.join(self.annotations_path, f"{image_id}.png")
        annotation = Image.open(annotation_path)  # Mask image where each pixel is a class label

        return image, annotation, image_id

class PC59Dataset(Dataset):
    def __init__(self, images_path, annotations_path, file_list_path):
        """
        Args:
            images_path (str): Path to the folder containing the image files (JPEG).
            annotations_path (str): Path to the folder containing the segmentation mask files (PNG).
            file_list_path (str): Path to the file containing the list of image IDs for validation.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.images_path = images_path
        self.annotations_path = annotations_path

        # Load the list of image IDs from the validation file
        with open(file_list_path, 'r') as file:
            self.file_list = file.read().splitlines()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Get the image ID from the validation list
        image_id = self.file_list[idx]

        # Load the image
        image_path = os.path.join(self.images_path, f"{image_id}.jpg")
        image = Image.open(image_path).convert("RGB")

        # Load the annotation (segmentation mask)
        annotation_path = os.path.join(self.annotations_path, f"{image_id}.png")
        annotation = Image.open(annotation_path)  # Each pixel is a class label

        return image, annotation, image_id


class PC459Dataset(Dataset):
    def __init__(self, images_path, annotations_path, file_list_path):
        """
        Args:
            images_path (str): Path to the folder containing the image files (JPEG).
            annotations_path (str): Path to the folder containing the segmentation mask files (TIF).
            file_list_path (str): Path to the file containing the list of image IDs for validation.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.images_path = images_path
        self.annotations_path = annotations_path

        # Load the list of image IDs from the validation file
        with open(file_list_path, 'r') as file:
            self.file_list = file.read().splitlines()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Get the image ID from the validation list
        image_id = self.file_list[idx]

        # Load the image
        image_path = os.path.join(self.images_path, f"{image_id}.jpg")
        image = Image.open(image_path).convert("RGB")

        # Load the annotation (segmentation mask in TIF format)
        annotation_path = os.path.join(self.annotations_path, f"{image_id}.tif")
        annotation = Image.open(annotation_path)  # Load TIF file (each pixel is a class label)

        return image, annotation, image_id