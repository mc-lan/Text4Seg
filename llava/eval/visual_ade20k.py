import numpy as np
import cv2
import os
import glob

# Define the new color palette for the classes
palette = [
    [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
    [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
    [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
    [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
    [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
    [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
    [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
    [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
    [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
    [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
    [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
    [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
    [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
    [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
    [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255]
]

# Define the classes
classes = (
    'background', 'aeroplane', 'bag', 'bed', 'bedclothes',
    'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle',
    'building', 'bus', 'cabinet', 'car', 'cat', 'ceiling',
    'chair', 'cloth', 'computer', 'cow', 'cup', 'curtain', 'dog',
    'door', 'fence', 'floor', 'flower', 'food', 'grass', 'ground',
    'horse', 'keyboard', 'light', 'motorbike', 'mountain',
    'mouse', 'person', 'plate', 'platform', 'pottedplant', 'road',
    'rock', 'sheep', 'shelves', 'sidewalk', 'sign', 'sky', 'snow',
    'sofa', 'table', 'track', 'train', 'tree', 'truck',
    'tvmonitor', 'wall', 'water', 'window', 'wood'
)

# Map classes to indices for easy access
label_names = {i: name for i, name in enumerate(classes)}

def load_image_and_mask(image_path, mask_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return image, mask

def convert_mask_to_rgb(mask):
    height, width = mask.shape
    rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for class_id, color in enumerate(palette):
        rgb_mask[mask == class_id] = color

    return rgb_mask

def apply_mask(image, rgb_mask, alpha=0.6):
    overlay_image = cv2.addWeighted(image, 1 - alpha, rgb_mask, alpha, 0)
    return overlay_image

def draw_legend(image, labels, palette):
    # Set the box dimensions
    box_height = 30
    start_y = image.shape[0] - box_height * len(labels) - 10
    
    for class_id, label in labels.items():
        if class_id < len(palette) and label != "background":  # Exclude "background"
            # Draw colored box
            cv2.rectangle(image, (10, start_y), (40, start_y + box_height), palette[class_id], -1)
            # Draw text
            cv2.putText(image, label, (50, start_y + box_height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            start_y += box_height  # Move down for the next entry

    return image

def get_present_labels(mask):
    unique_classes = np.unique(mask)
    present_labels = {cls: label_names[cls] for cls in unique_classes if cls in label_names and label_names[cls] != "background"}
    return present_labels

def process_images(base_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through each subdirectory in the base directory
    for subdir, _, _ in os.walk(base_dir):
        image_file = os.path.join(subdir, 'image.png')  # Assuming image file is named 'image.png'
        mask_file = os.path.join(subdir, 'sam_mask.png')  # Assuming mask file is named 'sam_mask.png'

        if not os.path.exists(image_file) or not os.path.exists(mask_file):
            print(f"Skipping {subdir}: image or mask not found.")
            continue
        
        # Load image and mask
        image, mask = load_image_and_mask(image_file, mask_file)
        
        # Convert mask to RGB
        rgb_mask = convert_mask_to_rgb(mask)
        
        # Apply mask to the image
        overlay_image = apply_mask(image, rgb_mask)
        
        # Get only present labels in the mask, excluding "background"
        present_labels = get_present_labels(mask)
        
        # Draw the legend on the overlay image
        overlay_image_with_legend = draw_legend(overlay_image, present_labels, palette)
        
        # Create output filename
        output_file = os.path.join(output_dir, os.path.basename(subdir) + '_overlay.png')
        cv2.imwrite(output_file, cv2.cvtColor(overlay_image_with_legend, cv2.COLOR_RGB2BGR))
        print(f"Saved overlay image to {output_file}")

# Set the base directory and output directory
base_directory = 'llava/eval/semantic_seg_results11/llava-v1.5-7b-lora-r64-coco-new/PC59'  # Update this to your actual path
output_directory = 'llava/eval/demo/PC59'  # Update with your desired output directory

# Process the images and masks
process_images(base_directory, output_directory)
