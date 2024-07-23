import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_transparent_mask(image_path, mask_path, transparency_level=100):
    """
    Apply a transparent mask to an image.

    Parameters:
    - image_path (str): The path to the input image.
    - mask_path (str): The path to the mask image.
    - transparency_level (int): The transparency level (0-255), where 0 is fully transparent and 255 is fully opaque.

    Returns:
    - rgba_image (numpy array): The image with the transparent mask applied.
    """
    # Load the images
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Check if images are loaded correctly
    if image is None:
        raise FileNotFoundError(f"Image file not found at {image_path}")
    if mask is None:
        raise FileNotFoundError(f"Mask file not found at {mask_path}")

    # Ensure the mask is binary
    _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    # Create an RGBA image from the original image
    rgba_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    # Set the alpha channel according to the mask
    rgba_image[:, :, 3] = np.where(binary_mask == 255, transparency_level,
                                   255)  # transparency_level is the transparency level

    return rgba_image


# Example usage
image_path = "C:/Users/Administrator/Projects/FORF/framework/data/Ground_Truth/SpotNeck.png"
mask_path = "C:/Users/Administrator/Projects/FORF/framework/data/Ground_Truth/SpotNeck_mask.png"
output_path = "C:/Users/Administrator/Projects/FORF/framework/data/BigLamaEvaluation/ImageAndMask/SpotNeck.png"

rgba_image = apply_transparent_mask(image_path, mask_path)

# Save the result
cv2.imwrite(output_path, rgba_image)

# Display the result
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(rgba_image, cv2.COLOR_BGRA2RGBA))
plt.axis('off')
plt.show()
