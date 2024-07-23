import cv2
import numpy as np


def inpaint_region_with_mask(image: np.ndarray, mask: np.ndarray, method: str = 'telea') -> np.ndarray:
    """
    Inpaint regions of an image specified by the mask using the specified method.

    Args:
        image (np.ndarray): The input image.
        mask (np.ndarray): The mask indicating regions to inpaint (non-zero values indicate the region).
        method (str): The inpainting method ('telea' or 'ns').

    Returns:
        np.ndarray: The inpainted image.
    """
    mask = (mask > 0).astype(np.uint8)
    if method == 'telea':
        inpainted_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    else:
        inpainted_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)
    return inpainted_image


def blur_region_with_mask(image: np.ndarray, mask: np.ndarray, blur_ksize: int = 15) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    blurred_image = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0)
    output_image = image.copy()
    output_image[mask == 1] = blurred_image[mask == 1]
    return output_image


def pixelate_region_with_mask(image: np.ndarray, mask: np.ndarray, pixel_size: int = 10) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    output_image = image.copy()
    for y in range(0, image.shape[0], pixel_size):
        for x in range(0, image.shape[1], pixel_size):
            if mask[y:y + pixel_size, x:x + pixel_size].any():
                pixel_region = image[y:y + pixel_size, x:x + pixel_size]
                color = pixel_region.mean(axis=(0, 1)).astype(int)
                output_image[y:y + pixel_size, x:x + pixel_size] = color
    return output_image


def apply_mask(image: np.ndarray, mask: np.ndarray, mask_color: tuple = (0, 0, 0)) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    output_image = image.copy()
    output_image[mask == 1] = mask_color
    return output_image


def distort_region_with_mask(image: np.ndarray, mask: np.ndarray, distortion_strength: float = 0.5) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    h, w = image.shape[:2]
    K = np.array([[w, 0, w // 2], [0, h, h // 2], [0, 0, 1]], dtype=np.float32)
    D = np.array([distortion_strength, distortion_strength, 0, 0], dtype=np.float32)

    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), 5)
    distorted_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)

    output_image = image.copy()
    output_image[mask == 1] = distorted_image[mask == 1]

    return output_image


def process_image_with_mask(image_path: str, mask_path: str, operation: str, **kwargs) -> np.ndarray:
    """
    Process an image with a given mask using a specified operation.

    Args:
        image_path (str): The path to the input image.
        mask_path (str): The path to the mask image.
        operation (str): The operation to apply ('blur', 'pixelate', 'mask', 'distort', 'inpaint').

    Returns:
        np.ndarray: The processed image.
    """
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Image file not found at {image_path}")
    if mask is None:
        raise FileNotFoundError(f"Mask file not found at {mask_path}")

    if operation == 'blur':
        return blur_region_with_mask(image, mask, **kwargs)
    elif operation == 'pixelate':
        return pixelate_region_with_mask(image, mask, **kwargs)
    elif operation == 'mask':
        return apply_mask(image, mask, **kwargs)
    elif operation == 'distort':
        return distort_region_with_mask(image, mask, **kwargs)
    elif operation == 'inpaint':
        return inpaint_region_with_mask(image, mask, **kwargs)
    else:
        raise ValueError("Unsupported operation. Choose from 'blur', 'pixelate', 'mask', 'distort', or 'inpaint'.")


# Example usage:
if __name__ == "__main__":
    image_path = '/framework/data/LaMa_Medical_Images/image2.png'
    mask_path = '/framework/data/LaMa_Medical_Images/image2_mask.png'

    try:
        # Choose the operation: 'blur', 'pixelate', 'mask', 'distort', or 'inpaint'
        result = process_image_with_mask(image_path, mask_path, operation='blur', blur_ksize=15)
        cv2.imwrite('blurred_image.png', result)

        result = process_image_with_mask(image_path, mask_path, operation='pixelate', pixel_size=10)
        cv2.imwrite('pixelated_image.png', result)

        result = process_image_with_mask(image_path, mask_path, operation='mask', mask_color=(0, 0, 0))
        cv2.imwrite('masked_image.png', result)

        result = process_image_with_mask(image_path, mask_path, operation='distort', distortion_strength=2)
        cv2.imwrite('distorted_image.png', result)

        result = process_image_with_mask(image_path, mask_path, operation='inpaint', method='telea')
        cv2.imwrite('inpainted_image.png', result)
    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)
