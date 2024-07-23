from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import numpy as np


def evaluate_inpainting(original_image_path, inpainted_image_path, mask_image_path):
    # Read the images
    original_image = imread(original_image_path)
    inpainted_image = imread(inpainted_image_path)
    mask = imread(mask_image_path, as_gray=True)

    # Convert mask to boolean
    mask = mask > 0

    # Print shapes for debugging
    print(f"Original image shape: {original_image.shape}")
    print(f"Inpainted image shape: {inpainted_image.shape}")
    print(f"Mask shape: {mask.shape}")

    # Ensure the mask has the same dimensions as the images
    if mask.shape != original_image.shape[:2] or mask.shape != inpainted_image.shape[:2]:
        raise ValueError("The mask dimensions do not match the image dimensions.")

    # Extract regions based on mask
    original_region = original_image[mask]
    inpainted_region = inpainted_image[mask]

    # Print shapes for debugging
    print(f"Original region shape: {original_region.shape}")
    print(f"Inpainted region shape: {inpainted_region.shape}")

    # Check if the original image is grayscale or color
    if len(original_image.shape) == 2:  # Grayscale
        original_region = np.stack((original_region,) * 3, axis=-1)  # Convert to 3 channels
    elif len(original_image.shape) == 3 and original_image.shape[2] == 3:  # Color
        pass
    else:
        raise ValueError("Unsupported image format.")

    # Print shapes after processing
    print(f"Processed original region shape: {original_region.shape}")

    # Calculate PSNR and SSIM
    psnr_value = psnr(original_region, inpainted_region)
    ssim_value = ssim(original_region, inpainted_region, multichannel=True)

    return {'PSNR': psnr_value, 'SSIM': ssim_value}


if __name__ == "__main__":
    evaluation_result = evaluate_inpainting(
        "/framework/data/Results/Inpainted_Images/image1_maskBIG.png",
        "/framework/data/Results/Inpainted_Images/image1_mask.png",
        "/framework/data/LaMa_Medical_Images/image1_mask.png"
    )
    print(evaluation_result)
