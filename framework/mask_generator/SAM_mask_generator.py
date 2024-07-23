import cv2
import subprocess
from tkinter import Tk, filedialog
from PIL import Image, ImageTk

from framework.mask_generator.config import Config


class SamSegmentGUI:
    def __init__(self, config):
        self.input_img = ""
        self.output_dir = ""
        self.point_coords = []
        self.sam_ckpt = config.sam_ckpt_path
        self.sam_segment_script = config.sam_segment_script
        self.sam_model_type = config.sam_model_type
        self.dilate_kernel_size = config.dilate_kernel_size
        self.point_labels = config.point_labels

    def select_image(self):
        self.input_img = filedialog.askopenfilename(title="Select Image")
        if not self.input_img:
            print("No image selected. Exiting.")
            exit()

    def select_output_directory(self):
        self.output_dir = filedialog.askdirectory(title="Select Output Directory")
        if not self.output_dir:
            print("No output directory selected. Exiting.")
            exit()

    def select_points_with_opencv(self):
        self.point_coords = get_clicked_point(self.input_img)
        if not self.point_coords:
            print("No points selected. Exiting.")
            exit()
        print(f"Point selected: {self.point_coords}")

    def run_sam_segment(self):
        if not self.point_coords:
            print("No points selected. Exiting.")
            exit()

        point_coords = [str(coord) for coord in self.point_coords]
        command = [
            "python", self.sam_segment_script,
            "--input_img", self.input_img,
            "--point_coords", *point_coords,
            "--point_labels", str(self.point_labels),
            "--dilate_kernel_size", str(self.dilate_kernel_size),
            "--output_dir", self.output_dir,
            "--sam_model_type", self.sam_model_type,
            "--sam_ckpt", self.sam_ckpt
        ]
        subprocess.run(command)

    def run(self):
        self.select_image()
        self.select_output_directory()
        self.select_points_with_opencv()
        self.run_sam_segment()


def click_event(event, x, y, flags, params):
    global point_coords
    if event == cv2.EVENT_LBUTTONDOWN:
        point_coords = (x, y)
        cv2.destroyAllWindows()


def get_clicked_point(img_path):
    img = cv2.imread(img_path)
    cv2.namedWindow("image")
    cv2.imshow("image", img)

    last_point = []
    keep_looping = True

    def mouse_callback(event, x, y, flags, param):
        nonlocal last_point, keep_looping, img

        if event == cv2.EVENT_LBUTTONDOWN:
            if last_point:
                cv2.circle(img, tuple(last_point), 5, (0, 0, 0), -1)
            last_point = [x, y]
            cv2.circle(img, tuple(last_point), 5, (0, 0, 255), -1)
            cv2.imshow("image", img)
        elif event == cv2.EVENT_RBUTTONDOWN:
            keep_looping = False

    cv2.setMouseCallback("image", mouse_callback)

    while keep_looping:
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    return last_point


if __name__ == "__main__":
    # Initialize configuration
    config = Config('config.json')

    app = SamSegmentGUI(config)
    app.run()

'''import warnings

import cv2
import torch
from segment_anything import SamPredictor, sam_model_registry
from skimage.io import imread, imsave
import numpy as np
from PIL import Image
from typing import List

# Global variables to store the point coordinates and labels
point_coords = []
point_labels = []


def click_event(event, x, y, flags, params):
    global point_coords, point_labels, image, predictor
    if event == cv2.EVENT_LBUTTONDOWN:
        # Add the clicked point
        point_coords.append([x, y])
        point_labels.append(1)

        # Update and display the mask
        update_and_display_mask()


def get_clicked_points(img_path):
    global image
    image = cv2.imread(img_path)
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_event)

    while True:
        cv2.imshow("image", image)
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' key to exit
            break

    cv2.destroyAllWindows()


def update_and_display_mask():
    global point_coords, point_labels, image, predictor

    if len(point_coords) == 0:
        return

    # Predict the mask using SAM
    masks, _, _ = predictor.predict(
        point_coords=np.array(point_coords),
        point_labels=np.array(point_labels),
        multimask_output=False
    )

    # Extract and process the mask
    mask = masks[0]
    mask = mask.astype(np.uint8) * 255

    # Display the mask on the image
    mask_display = cv2.addWeighted(image, 1, np.stack([mask] * 3, axis=-1), 0.5, 0)
    cv2.imshow("image", mask_display)


def generate_mask_interactive(input_img_path, output_mask_path, sam_model_type="vit_h",
                              sam_ckpt="C:/Users/Administrator/Projects/FORF/Inpaint-Anything/pretrained_models/sam_vit_h_4b8939.pth"):
    global image, predictor

    # Load the image
    image = load_img_to_array(input_img_path)

    # Ensure image is 3D
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Load SAM model
    sam = sam_model_registry[sam_model_type](checkpoint=sam_ckpt)
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    # Get the clicked points
    get_clicked_points(input_img_path)

    # Save the final mask
    if len(point_coords) > 0:
        masks, _, _ = predictor.predict(
            point_coords=np.array(point_coords),
            point_labels=np.array(point_labels),
            multimask_output=False
        )

        mask = masks[0]
        mask = mask.astype(np.uint8) * 255

        # Ensure the mask is 2D
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # Suppress the low contrast warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            imsave(output_mask_path, mask)

        print(f"Mask saved to {output_mask_path}")
    else:
        print("No points selected. No mask generated.")


def load_img_to_array(img_p):
    img = Image.open(img_p)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    return np.array(img)
'''

'''import warnings

import cv2
import torch
from segment_anything import SamPredictor, sam_model_registry
from skimage.io import imread, imsave
import numpy as np
from PIL import Image
from typing import List

# Global variable to store the point coordinates
point_coords = []


def click_event(event, x, y, flags, params):
    global point_coords
    if event == cv2.EVENT_LBUTTONDOWN:
        point_coords = (x, y)
        cv2.destroyAllWindows()


def get_clicked_point(img_path):
    img = cv2.imread(img_path)
    cv2.namedWindow("image")
    cv2.imshow("image", img)

    last_point = []
    keep_looping = True

    def mouse_callback(event, x, y, flags, param):
        nonlocal last_point, keep_looping, img

        if event == cv2.EVENT_LBUTTONDOWN:
            if last_point:
                cv2.circle(img, tuple(last_point), 5, (0, 0, 0), -1)
            last_point = [x, y]
            cv2.circle(img, tuple(last_point), 5, (0, 0, 255), -1)
            cv2.imshow("image", img)
        elif event == cv2.EVENT_RBUTTONDOWN:
            keep_looping = False

    cv2.setMouseCallback("image", mouse_callback)

    while keep_looping:
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    return last_point


def predict_masks_with_sam(img: np.ndarray, point_coords: List[List[float]], point_labels: List[int], model_type: str,
                           ckpt_p: str, device="cuda"):
    point_coords = np.array(point_coords)
    point_labels = np.array(point_labels)
    sam = sam_model_registry[model_type](checkpoint=ckpt_p)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    predictor.set_image(img)
    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )
    return masks, scores, logits


def generate_mask_with_click(input_img_path, output_mask_path, sam_model_type="vit_h",
                             sam_ckpt="C:/Users/Administrator/Projects/FORF/Inpaint-Anything/pretrained_models/sam_vit_h_4b8939.pth"):
    global point_coords

    # Load the image
    image = load_img_to_array(input_img_path)

    # Ensure image is 3D
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Get the clicked point
    point_coords = get_clicked_point(input_img_path)

    if not point_coords:
        print("No point selected. Exiting.")
        return

    print(f"Selected coordinates: {point_coords}")

    # Predict the mask using SAM
    masks, _, _ = predict_masks_with_sam(
        image,
        [point_coords],
        [1],  # Label for the point
        model_type=sam_model_type,
        ckpt_p=sam_ckpt,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Extract and process the mask
    mask = masks[0]
    mask = mask[0] if len(mask.shape) == 3 else mask
    mask = mask.astype(np.uint8) * 255

    # Ensure the mask is 2D
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    # Suppress the low contrast warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        imsave(output_mask_path, mask)


def load_img_to_array(img_p):
    img = Image.open(img_p)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    return np.array(img)

'''
