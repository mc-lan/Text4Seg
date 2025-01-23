import random

def encode_mask(mask_list):
    rows = []
    for row in mask_list:
        encoded_row = []
        count = 1
        for j in range(1, len(row)):
            if row[j] == row[j - 1]:
                count += 1
            else:
                encoded_row.append(f"{row[j - 1]} *{count}")
                count = 1
        encoded_row.append(f"{row[-1]} *{count}")
        rows.append("| ".join(encoded_row))
    return "\n ".join(rows) + "\n"


def random_crop(image):
    width, height = image.size

    # List of available crop sizes
    crop_sizes = [256, 336, 392, 448]
    
    # Filter the crop sizes to only include those smaller than the image dimensions
    valid_crop_sizes = [size for size in crop_sizes if size <= width and size <= height]

    # If no valid crop sizes are available, raise an error
    if not valid_crop_sizes:
        valid_crop_sizes = [min(width, height)]
    
    # Randomly choose a valid crop size
    crop_size = random.choice(valid_crop_sizes)

    # Randomly choose the crop's top-left corner
    left = random.randint(0, width - crop_size)
    top = random.randint(0, height - crop_size)

    # Calculate the right and bottom coordinates based on the crop size
    right = left + crop_size
    bottom = top + crop_size

    # Crop the image
    cropped_image = image.crop((left, top, right, bottom))

    # Return the crop coordinates and the cropped image
    return cropped_image, (left, top, right, bottom)