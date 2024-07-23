import pydicom
import numpy as np
import cv2


def read_dicom(dicom_path):
    """
    Reads a DICOM file and converts its pixel data to a normalized 8-bit grayscale image.

    Parameters:
    - dicom_path (str): The path to the DICOM file to be read.

    Returns:
    - np.ndarray: The normalized 8-bit grayscale image.
    """
    dicom = pydicom.dcmread(dicom_path)
    img = dicom.pixel_array
    img = img - np.min(img)
    img = img / np.max(img)
    img = (img * 255).astype(np.uint8)
    return img

def dicom_to_png(dicom_path, output_path):
    """
    Converts a DICOM file to a PNG file.

    Parameters:
    - dicom_path (str): The path to the DICOM file to be converted.
    - output_path (str): The path where the converted PNG file will be saved.

    Returns:
    - str: The path to the saved PNG file.
    """
    img = read_dicom(dicom_path)
    cv2.imwrite(output_path, img)
    return output_path


def save_image_to_dicom(img, metadata, output_dicom_path):
    """
    Saves an image to a DICOM file with the provided metadata.

    Parameters:
    - img (np.ndarray): The image to be saved in the DICOM file.
    - metadata (pydicom.dataset.FileDataset): The metadata of the original DICOM file to be retained.
    - output_dicom_path (str): The path where the output DICOM file will be saved.

    Returns:
    - None
    """
    img = img.astype(metadata.pixel_array.dtype)
    metadata.PixelData = img.tobytes()
    metadata.save_as(output_dicom_path)