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
    cv2.namedWindow("image / Right click to continue")
    cv2.imshow("image / Right click to continue", img)

    last_point = []
    keep_looping = True

    def mouse_callback(event, x, y, flags, param):
        nonlocal last_point, keep_looping, img

        if event == cv2.EVENT_LBUTTONDOWN:
            if last_point:
                cv2.circle(img, tuple(last_point), 5, (0, 0, 0), -1)
            last_point = [x, y]
            cv2.circle(img, tuple(last_point), 5, (0, 0, 255), -1)
            cv2.imshow("image / Right click to continue", img)
        elif event == cv2.EVENT_RBUTTONDOWN:
            keep_looping = False

    cv2.setMouseCallback("image / Right click to continue", mouse_callback)


    while keep_looping:
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    return last_point


if __name__ == "__main__":
    # Initialize configuration
    config = Config('config.json')

    app = SamSegmentGUI(config)
    app.run()


