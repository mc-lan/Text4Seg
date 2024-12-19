import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the original image and mask
image_path = "images/hhh.jpg"
mask_path = "images/horse_mask.png"

# Read the image and the mask
image = cv2.imread(image_path)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load the mask as grayscale

# Convert the image from BGR to RGB for visualization with Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Normalize the mask to [0, 1] for transparency
mask_normalized = mask

# Create a colored mask (red color in this case, you can change it)
colored_mask = np.zeros_like(image_rgb)
colored_mask[:, :, 0] = mask_normalized * 255  # Applying the mask to the red channel

# Create the overlay image by blending the original image and colored mask
alpha = 0.6  # Transparency factor
overlay_image = image_rgb.copy()

# Only apply the mask where it is not zero (non-background areas)
overlay_image[mask > 0] = cv2.addWeighted(image_rgb[mask > 0], 1 - alpha, colored_mask[mask > 0], alpha, 0)

# Save the final image
output_path = "images/combined_image.png"
cv2.imwrite(output_path, cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))

# Display the combined image
plt.figure(figsize=(6, 6))
plt.imshow(overlay_image)
plt.title("Image with Mask Overlay")
plt.axis("off")
plt.show()
