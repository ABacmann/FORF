import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog, Toplevel
from PIL import Image, ImageTk
import os


class MaskGenerator:
    def __init__(self, image_path, output_dir):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Failed to load image at {image_path}")
        self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.brush_size = 30
        self.drawing = False
        self.image_display = self.image.copy()
        self.output_dir = output_dir
        self.image_name = os.path.basename(image_path)
        self.photo = None
        self.canvas_image = None
        self.zoom_factor = 1.0
        self.offset_x, self.offset_y = 0, 0
        self.pan_start_x, self.pan_start_y = 0, 0
        self.panning = False
        self.update_scheduled = False  # Track if an update is already scheduled

        self.setup_gui()

    def setup_gui(self):
        self.root = Toplevel()
        self.root.title(f"Mask Generator - {self.image_name}")

        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas_frame = tk.Frame(main_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scroll_x = tk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.scroll_y = tk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.configure(xscrollcommand=self.scroll_x.set, yscrollcommand=self.scroll_y.set)

        self.photo = self.convert_image_for_tkinter(self.image_display)
        if self.photo is not None:
            self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
            print(f"Successfully loaded and displayed image {self.image_name}")
        else:
            print(f"Failed to load image {self.image_name} onto the canvas")

        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        self.canvas.bind("<MouseWheel>", self.zoom)

        self.canvas.bind("<ButtonPress-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.pan)
        self.canvas.bind("<ButtonRelease-2>", self.stop_pan)

        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.brush_size_button = tk.Button(button_frame, text="Brush Size", command=self.change_brush_size)
        self.brush_size_button.pack(side=tk.LEFT)

        self.save_button = tk.Button(button_frame, text="Save Mask", command=self.save_mask)
        self.save_button.pack(side=tk.RIGHT)

        self.done_button = tk.Button(button_frame, text="Done", command=self.done)
        self.done_button.pack(side=tk.RIGHT)

    def convert_image_for_tkinter(self, image):
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            return ImageTk.PhotoImage(image_pil)
        except Exception as e:
            print(f"Error converting image to PhotoImage: {e}")
            return None

    def start_draw(self, event):
        self.drawing = True
        self.last_x, self.last_y = self.canvas_to_image_coords(event.x, event.y)

    def draw(self, event):
        if self.drawing:
            current_x, current_y = self.canvas_to_image_coords(event.x, event.y)
            cv2.line(self.image_display, (self.last_x, self.last_y), (current_x, current_y), (255, 255, 255),
                     self.brush_size)
            cv2.line(self.mask, (self.last_x, self.last_y), (current_x, current_y), 255, self.brush_size)
            self.last_x, self.last_y = current_x, current_y
            self.schedule_canvas_update()

    def stop_draw(self, event):
        self.drawing = False

    def start_pan(self, event):
        self.panning = True
        self.pan_start_x, self.pan_start_y = event.x, event.y

    def pan(self, event):
        if self.panning:
            dx = event.x - self.pan_start_x
            dy = event.y - self.pan_start_y
            self.offset_x += dx
            self.offset_y += dy
            self.pan_start_x, self.pan_start_y = event.x, event.y
            self.schedule_canvas_update()

    def stop_pan(self, event):
        self.panning = False

    def canvas_to_image_coords(self, x, y):
        x = int((x - self.offset_x) / self.zoom_factor)
        y = int((y - self.offset_y) / self.zoom_factor)
        return x, y

    def schedule_canvas_update(self):
        if not self.update_scheduled:
            self.update_scheduled = True
            self.canvas.after_idle(self.update_canvas)

    def update_canvas(self):
        height, width = self.image.shape[:2]
        resized_image = cv2.resize(self.image_display, (int(width * self.zoom_factor), int(height * self.zoom_factor)))
        new_photo = self.convert_image_for_tkinter(resized_image)
        if new_photo is not None:
            self.photo = new_photo
            if self.canvas_image is not None:
                self.canvas.itemconfig(self.canvas_image, image=self.photo)
                self.canvas.coords(self.canvas_image, self.offset_x, self.offset_y)
                self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
                self.canvas.update()
        self.update_scheduled = False

    def zoom(self, event):
        factor = 1.1 if event.delta > 0 else 0.9
        self.zoom_factor *= factor
        self.schedule_canvas_update()

    def change_brush_size(self):
        self.brush_size = simpledialog.askinteger("Brush Size", "Enter brush size:", initialvalue=self.brush_size)

    def save_mask(self):
        base_name, ext = os.path.splitext(self.image_name)
        mask_name = f"{base_name}_mask.png"
        save_path = os.path.join(self.output_dir, mask_name)
        cv2.imwrite(save_path, self.mask)
        print(f"Mask saved at {save_path}")

    def done(self):
        self.save_mask()
        self.root.destroy()



# To test the MaskGenerator class with the provided image
if __name__ == "__main__":
    # Test with the provided image
    test_image_path = "/framework/data/LaMa_Medical_Images/ChestXRayPacemaker.png"  # Adjust the path to your uploaded image
    test_output_dir = "C:/Users/Administrator/Projects/ForeignObjectRemovalFrameworkBalgrist/Data/Results/Masks"  # Define an output directory

    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir)

    # Create a mask for the test image
    MaskGenerator(test_image_path, test_output_dir)

'''import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog, Toplevel
from PIL import Image, ImageTk
import os


class MaskGenerator:
    def __init__(self, image_path, output_dir):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Failed to load image at {image_path}")
        self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.brush_size = 30
        self.drawing = False
        self.image_display = self.image.copy()
        self.output_dir = output_dir
        self.image_name = os.path.basename(image_path)
        self.photo = None
        self.canvas_image = None
        self.zoom_factor = 1.0
        self.offset_x, self.offset_y = 0, 0
        self.pan_start_x, self.pan_start_y = 0, 0
        self.panning = False
        self.update_scheduled = False  # Track if an update is already scheduled

        self.setup_gui()

    def setup_gui(self):
        self.root = Toplevel()
        self.root.title(f"Mask Generator - {self.image_name}")

        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas_frame = tk.Frame(main_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scroll_x = tk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.scroll_y = tk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.configure(xscrollcommand=self.scroll_x.set, yscrollcommand=self.scroll_y.set)

        self.photo = self.convert_image_for_tkinter(self.image_display)
        if self.photo is not None:
            self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
            print(f"Successfully loaded and displayed image {self.image_name}")
        else:
            print(f"Failed to load image {self.image_name} onto the canvas")

        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        self.canvas.bind("<MouseWheel>", self.zoom)

        self.canvas.bind("<ButtonPress-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.pan)
        self.canvas.bind("<ButtonRelease-2>", self.stop_pan)

        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.brush_size_button = tk.Button(button_frame, text="Brush Size", command=self.change_brush_size)
        self.brush_size_button.pack(side=tk.LEFT)

        self.save_button = tk.Button(button_frame, text="Save Mask", command=self.save_mask)
        self.save_button.pack(side=tk.RIGHT)

    def convert_image_for_tkinter(self, image):
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            return ImageTk.PhotoImage(image_pil)
        except Exception as e:
            print(f"Error converting image to PhotoImage: {e}")
            return None

    def start_draw(self, event):
        self.drawing = True
        self.last_x, self.last_y = self.canvas_to_image_coords(event.x, event.y)

    def draw(self, event):
        if self.drawing:
            current_x, current_y = self.canvas_to_image_coords(event.x, event.y)
            cv2.line(self.image_display, (self.last_x, self.last_y), (current_x, current_y), (255, 255, 255),
                     self.brush_size)
            cv2.line(self.mask, (self.last_x, self.last_y), (current_x, current_y), 255, self.brush_size)
            self.last_x, self.last_y = current_x, current_y
            self.schedule_canvas_update()

    def stop_draw(self, event):
        self.drawing = False

    def start_pan(self, event):
        self.panning = True
        self.pan_start_x, self.pan_start_y = event.x, event.y

    def pan(self, event):
        if self.panning:
            dx = event.x - self.pan_start_x
            dy = event.y - self.pan_start_y
            self.offset_x += dx
            self.offset_y += dy
            self.pan_start_x, self.pan_start_y = event.x, event.y
            self.schedule_canvas_update()

    def stop_pan(self, event):
        self.panning = False

    def canvas_to_image_coords(self, x, y):
        x = int((x - self.offset_x) / self.zoom_factor)
        y = int((y - self.offset_y) / self.zoom_factor)
        return x, y

    def schedule_canvas_update(self):
        if not self.update_scheduled:
            self.update_scheduled = True
            self.canvas.after_idle(self.update_canvas)

    def update_canvas(self):
        height, width = self.image.shape[:2]
        resized_image = cv2.resize(self.image_display, (int(width * self.zoom_factor), int(height * self.zoom_factor)))
        new_photo = self.convert_image_for_tkinter(resized_image)
        if new_photo is not None:
            self.photo = new_photo
            if self.canvas_image is not None:
                self.canvas.itemconfig(self.canvas_image, image=self.photo)
                self.canvas.coords(self.canvas_image, self.offset_x, self.offset_y)
                self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
                self.canvas.update()
        self.update_scheduled = False

    def zoom(self, event):
        factor = 1.1 if event.delta > 0 else 0.9
        self.zoom_factor *= factor
        self.schedule_canvas_update()

    def change_brush_size(self):
        self.brush_size = simpledialog.askinteger("Brush Size", "Enter brush size:", initialvalue=self.brush_size)

    def save_mask(self):
        base_name, ext = os.path.splitext(self.image_name)
        mask_name = f"{base_name}_mask.png"
        save_path = os.path.join(self.output_dir, mask_name)
        cv2.imwrite(save_path, self.mask)
        print(f"Mask saved at {save_path}")'''