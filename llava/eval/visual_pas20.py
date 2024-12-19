import numpy as np
import cv2
import os
import glob

# Define the color palette for the classes
palette = [
    [128, 0, 0], [0, 128, 0], [0, 0, 192],
    [128, 128, 0], [128, 0, 128], [0, 128, 128],
    [192, 128, 64], [64, 0, 0], [192, 0, 0],
    [64, 128, 0], [192, 128, 0], [64, 0, 128],
    [192, 0, 128], [64, 128, 128], [192, 128, 128],
    [0, 64, 0], [128, 64, 0], [0, 192, 0],
    [128, 192, 0], [0, 64, 128]
]

# Define label names corresponding to each index
label_names = {
    0: "others",
    1: "aeroplane",
    2: "bicycle",
    3: "bird",
    4: "boat",
    5: "bottle",
    6: "bus",
    7: "car",
    8: "cat",
    9: "chair",
    10: "cow",
    11: "dining table",
    12: "dog",
    13: "horse",
    14: "motorbike",
    15: "person",
    16: "pottedplant",
    17: "sheep",
    18: "sofa",
    19: "train",
    20: "TV"
}

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
        if class_id < len(palette) and label != "others":  # Exclude "others"
            # Draw colored box
            cv2.rectangle(image, (10, start_y), (40, start_y + box_height), palette[class_id], -1)
            # Draw text
            cv2.putText(image, label, (50, start_y + box_height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            start_y += box_height  # Move down for the next entry

    return image

def get_present_labels(mask):
    unique_classes = np.unique(mask)
    present_labels = {cls: label_names[cls] for cls in unique_classes if cls in label_names and label_names[cls] != "others"}
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
        
        # Get only present labels in the mask, excluding "others"
        present_labels = get_present_labels(mask)
        
        # Draw the legend on the overlay image
        overlay_image_with_legend = draw_legend(overlay_image, present_labels, palette)
        
        # Create output filename
        output_file = os.path.join(output_dir, os.path.basename(subdir) + '_overlay.png')
        cv2.imwrite(output_file, cv2.cvtColor(overlay_image_with_legend, cv2.COLOR_RGB2BGR))
        print(f"Saved overlay image to {output_file}")

# Set the base directory and output directory
base_directory = 'llava/eval/semantic_seg_results11/llava-v1.5-7b-lora-r64-coco-new/PAS20'  # Update this to your actual path
output_directory = 'llava/eval/demo/PAS20'  # Update with your desired output directory

# Process the images and masks
process_images(base_directory, output_directory)
