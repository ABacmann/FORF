import os
import shutil
from tkinter import filedialog, messagebox, Toplevel, Label, Button, Frame, Tk
from PIL import Image, ImageTk
import cv2
from pathlib import Path
from framework.inpainting_pipeline import InpaintingManager
from framework.models.Big_Lama_Fourier import BigLamaModel, FineTunedLamaModel, FineTunedLamaModel2
from framework.models.trivial_removal import process_image_with_mask
from framework.mask_generator.manual_mask_generator import MaskGenerator
from framework.models.Deep_Fill_V2 import DeepFillV2
from framework.mask_generator.config import Config
from framework.mask_generator.SAM_mask_generator import SamSegmentGUI  # Import the SAM-based mask generator

# Initialize configuration
from framework.utils import dicom_to_png

config = Config('framework/mask_generator/config.json')


class FileHandler:
    @staticmethod
    def create_output_directory(base_dir, sub_dir):
        """
        Creates a directory (if it doesn't exist) and returns its path.
        """
        output_dir = os.path.join(base_dir, sub_dir)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Directory created or exists: {output_dir}")
        return output_dir

    @staticmethod
    def setup_output_directories(base_output_dir):
        """
        Sets up the required output directories and returns their paths.
        """
        mask_output_dir = FileHandler.create_output_directory(base_output_dir, "Masks")
        inpaint_output_dir = FileHandler.create_output_directory(base_output_dir, "Inpainted_Images")
        final_input_dir = FileHandler.create_output_directory(base_output_dir, "Ground_Truth")
        return mask_output_dir, inpaint_output_dir, final_input_dir

    @staticmethod
    def choose_files_or_folder(root, image_paths):
        """
        Opens a dialog to select files or a folder containing images and updates the image_paths list.
        """

        def select_files():
            files = filedialog.askopenfilenames(
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.dcm"), ("All files", "*.*")])
            image_paths.extend(files)
            root.quit()

        def select_folder():
            folder_path = filedialog.askdirectory(title="Select Folder with Images")
            for file_name in os.listdir(folder_path):
                if file_name.endswith(('.png', '.jpg', '.jpeg', '.dcm')):
                    image_paths.append(os.path.join(folder_path, file_name))
            root.quit()

        selection_window = Toplevel(root)
        selection_window.title("Select Input Type")

        frame = Frame(selection_window)
        frame.pack(pady=20, padx=20)

        file_button = Button(frame, text="Select Files", command=select_files, width=20)
        file_button.grid(row=0, column=0, pady=10)

        folder_button = Button(frame, text="Select Folder", command=select_folder, width=20)
        folder_button.grid(row=1, column=0, pady=10)

        selection_window.mainloop()

    @staticmethod
    def convert_images_to_png(image_paths, final_input_dir):
        """
        Converts images to PNG format and saves them in the specified directory.
        """
        png_paths = []
        for image_path in image_paths:
            base_name = os.path.basename(image_path)
            name, _ = os.path.splitext(base_name)
            png_path = os.path.join(final_input_dir, f"{name}.png")

            try:
                if image_path.lower().endswith('.dcm'):
                    dicom_to_png(image_path, png_path)
                    png_paths.append(png_path)
                elif image_path.lower().endswith(('.jpg', '.jpeg')):
                    img = Image.open(image_path)
                    img.save(png_path)
                    png_paths.append(png_path)
                else:
                    png_paths.append(image_path)
            except Exception as e:
                print(f"Error converting {image_path} to PNG: {e}")

        return png_paths


class MaskGeneratorHandler:
    @staticmethod
    def generate_manual_mask(image_path, mask_output_dir):
        """
        Generates a manual mask for the given image and saves it to the specified directory.
        """
        try:
            mask_gen = MaskGenerator(image_path, mask_output_dir)
            mask_gen.root.wait_window(mask_gen.root)
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            return os.path.join(mask_output_dir, f"{name}_mask.png")
        except Exception as e:
            print(f"Error generating manual mask for {image_path}: {e}")
            return None

    @staticmethod
    def generate_sam_mask(image_path, mask_output_dir):
        """
        Generates a SAM mask for the given image using a segmentation GUI and saves it to the specified directory.
        """
        try:
            sam_segment_gui = SamSegmentGUI(config)
            sam_segment_gui.input_img = image_path
            sam_segment_gui.output_dir = mask_output_dir
            sam_segment_gui.select_points_with_opencv()
            sam_segment_gui.run_sam_segment()
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            masks_dir = os.path.join(mask_output_dir, name)
            masks = [os.path.join(masks_dir, f"mask_{i}.png") for i in range(3)]
            visual_masks = [os.path.join(masks_dir, f"with_mask_{i}.png") for i in range(3)]
            return masks, visual_masks
        except Exception as e:
            print(f"Error generating SAM mask for {image_path}: {e}")
            return [], []

    @staticmethod
    def copy_generated_mask(generated_mask_path, final_input_dir, new_name):
        """
        Copies the generated mask to the final input directory with a new name.
        """
        try:
            if os.path.exists(generated_mask_path):
                destination = os.path.join(final_input_dir, new_name)
                shutil.copy(generated_mask_path, destination)
                print(f"Copied mask to: {destination}")
            else:
                print(f"Mask not found: {generated_mask_path}")
        except Exception as e:
            print(f"Error copying mask {generated_mask_path}: {e}")

    @staticmethod
    def choose_mask(visual_masks):
        """
        Opens a window to allow the user to choose one of the visual masks.
        """
        root = Toplevel()
        root.title("Choose a Mask")
        chosen_mask = [None]

        def select_mask(mask_index):
            chosen_mask[0] = mask_index
            root.quit()

        try:
            for i, visual_mask in enumerate(visual_masks):
                img = Image.open(visual_mask)
                img = img.resize((300, 300))
                img = ImageTk.PhotoImage(img)
                panel = Label(root, image=img)
                panel.image = img
                panel.grid(row=0, column=i)
                button = Button(root, text=f"Select Mask {i}", command=lambda i=i: select_mask(i))
                button.grid(row=1, column=i)
        except Exception as e:
            print(f"Error displaying mask selection: {e}")

        root.mainloop()
        root.destroy()
        return chosen_mask[0]


class InpaintingHandler:
    @staticmethod
    def process_single_image(image_path, mask_output_dir, final_input_dir, mask_method):
        """
        Processes a single image: copies it to the final input directory and generates a mask using the specified method.
        """
        try:
            print(f"Processing single image: {image_path}")

            destination_path = Path(final_input_dir) / Path(image_path).name
            if Path(image_path).resolve() != destination_path.resolve():
                shutil.copy(image_path, final_input_dir)
                print(f"Copied image to final input dir: {final_input_dir}")

            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            if mask_method == 'manual':
                generated_mask_path = MaskGeneratorHandler.generate_manual_mask(image_path, mask_output_dir)
                if generated_mask_path:
                    print(f"Generated manual mask: {generated_mask_path}")
                    MaskGeneratorHandler.copy_generated_mask(generated_mask_path, final_input_dir, f"{name}_mask.png")
            else:
                generated_masks, visual_masks = MaskGeneratorHandler.generate_sam_mask(image_path, mask_output_dir)
                if generated_masks and visual_masks:
                    print(f"Generated SAM masks: {generated_masks}")
                    chosen_mask_index = MaskGeneratorHandler.choose_mask(visual_masks)
                    chosen_mask = generated_masks[chosen_mask_index]
                    print(f"Chosen mask: {chosen_mask}")
                    MaskGeneratorHandler.copy_generated_mask(chosen_mask, final_input_dir, f"{name}_mask.png")
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    @staticmethod
    def process_multiple_images(image_paths, mask_output_dir, final_input_dir, mask_method):
        """
        Processes multiple images: copies each to the final input directory and generates masks using the specified method.
        """
        for image_path in image_paths:
            try:
                shutil.copy(image_path, final_input_dir)

                base_name = os.path.basename(image_path)
                name, ext = os.path.splitext(base_name)
                if mask_method == 'manual':
                    generated_mask_path = MaskGeneratorHandler.generate_manual_mask(image_path, mask_output_dir)
                    if generated_mask_path:
                        MaskGeneratorHandler.copy_generated_mask(generated_mask_path, final_input_dir,
                                                                 f"{name}_mask.png")
                else:
                    generated_masks, visual_masks = MaskGeneratorHandler.generate_sam_mask(image_path, mask_output_dir)
                    if generated_masks and visual_masks:
                        chosen_mask_index = MaskGeneratorHandler.choose_mask(visual_masks)
                        chosen_mask = generated_masks[chosen_mask_index]
                        MaskGeneratorHandler.copy_generated_mask(chosen_mask, final_input_dir, f"{name}_mask.png")
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")


class PipelineManager:
    @staticmethod
    def ask_yes_no_question(root, question, result):
        """
        Asks a yes/no question using a message box and appends the answer to the result list.
        """
        choice = messagebox.askquestion("Question", question)
        result.append('yes' if choice == 'yes' else 'no')

    @staticmethod
    def select_mask_generation_method(root):
        """
        Opens a dialog for the user to select a mask generation method (manual or SAM).
        """
        methods = ['manual', 'SAM']
        selected_method = [None]

        def set_method(method_name):
            selected_method[0] = method_name
            root.quit()

        method_window = Toplevel(root)
        method_window.title("Select Mask Generation Method")

        frame = Frame(method_window)
        frame.pack(pady=20, padx=20)

        for method in methods:
            button = Button(frame, text=method, command=lambda m=method: set_method(m), width=20)
            button.pack(pady=5)

        method_window.mainloop()
        return selected_method[0]

    @staticmethod
    def select_model_manually(root):
        """
        Opens a dialog for the user to select an inpainting model from a predefined list.
        """
        models = ['DeepFillV2', 'BigLamaModel', 'FineTunedLamaModel', 'FineTunedLamaModel2']
        selected_model = [None]

        def set_model(model_name):
            selected_model[0] = model_name
            root.quit()

        model_window = Toplevel(root)
        model_window.title("Select Inpainting Model")

        frame = Frame(model_window)
        frame.pack(pady=20, padx=20)

        for model in models:
            button = Button(frame, text=model, command=lambda m=model: set_model(m), width=20)
            button.pack(pady=5)

        model_window.mainloop()
        return selected_model[0]

    @staticmethod
    def select_inpainting_model(selected_model_name):
        """
        Selects and returns an instance of the inpainting model based on the selected model name.
        """
        models_with_torch_home = {
            'BigLamaModel': BigLamaModel,
            'FineTunedLamaModel': FineTunedLamaModel,
            'FineTunedLamaModel2': FineTunedLamaModel2
        }

        if selected_model_name == 'DeepFillV2':
            return DeepFillV2()

        model_class = models_with_torch_home.get(selected_model_name)
        if model_class:
            try:
                return model_class(torch_home=os.getenv('TORCH_HOME'))
            except Exception as e:
                print(f"Error initializing model {selected_model_name}: {e}")
                return None
        else:
            print(f"Unknown model selected: {selected_model_name}")
            return None

    @staticmethod
    def select_removal_method(root):
        """
        Opens a dialog for the user to select an image removal method from a predefined list.
        """
        methods = ['blur', 'pixelate', 'mask', 'distort', 'inpaint']
        selected_method = [None]

        def set_method(method_name):
            selected_method[0] = method_name
            root.quit()

        method_window = Toplevel(root)
        method_window.title("Select Removal Method")

        frame = Frame(method_window)
        frame.pack(pady=20, padx=20)

        for method in methods:
            button = Button(frame, text=method, command=lambda m=method: set_method(m), width=20)
            button.pack(pady=5)

        method_window.mainloop()
        return selected_method[0]

    @staticmethod
    def manage_multiple_images_without_masks(root, image_paths, mask_output_dir, final_input_dir):
        """
        Manages the processing of multiple images without existing masks by selecting a mask generation method.
        """
        mask_method = PipelineManager.select_mask_generation_method(root)
        if mask_method:
            InpaintingHandler.process_multiple_images(image_paths, mask_output_dir, final_input_dir, mask_method)
        else:
            print("No mask generation method selected. Exiting.")

    @staticmethod
    def manage_multiple_images_with_masks(image_paths, final_input_dir):
        """
        Manages the processing of multiple images with existing masks, copying them to the final input directory.
        """
        for image_path in image_paths:
            try:
                destination_path = Path(final_input_dir) / Path(image_path).name
                if Path(image_path).resolve() != destination_path.resolve():
                    shutil.copy(image_path, final_input_dir)
                    print(f"Copied image to final input dir: {final_input_dir}")

                base_name = os.path.basename(image_path)
                name, ext = os.path.splitext(base_name)
                mask_path = os.path.join(os.path.dirname(image_path), f"{name}_mask.png")
                if os.path.exists(mask_path):
                    destination_mask_path = Path(final_input_dir) / Path(mask_path).name
                    if Path(mask_path).resolve() != destination_mask_path.resolve():
                        shutil.copy(mask_path, final_input_dir)
                        print(f"Copied mask to final input dir: {final_input_dir}")
                else:
                    print(f"No mask found for {image_path}")
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")

    @staticmethod
    def manage_single_image(root, image_paths, mask_output_dir, final_input_dir):
        """
        Manages the processing of a single image by selecting a mask generation method.
        """
        mask_method = PipelineManager.select_mask_generation_method(root)
        if mask_method:
            InpaintingHandler.process_single_image(image_paths[0], mask_output_dir, final_input_dir, mask_method)
        else:
            print("No mask generation method selected. Exiting.")

    @staticmethod
    def use_inpainting_method(root, final_input_dir, inpaint_output_dir):
        """
        Uses the selected inpainting model to process images in the final input directory and save results in the output directory.
        """
        selected_model_name = PipelineManager.select_model_manually(root)
        inpainting_model = PipelineManager.select_inpainting_model(selected_model_name)
        if inpainting_model:
            try:
                pipeline = InpaintingManager(inpainting_model)
                pipeline.process_directory(final_input_dir, inpaint_output_dir)
            except Exception as e:
                print(f"Error running inpainting pipeline: {e}")

    @staticmethod
    def use_trivial_anonymization_method(root, image_paths, final_input_dir, inpaint_output_dir):
        """
        Applies a selected trivial anonymization method (e.g., blur, pixelate) to the images.
        """
        selected_method = PipelineManager.select_removal_method(root)
        if selected_method:
            for image_path in image_paths:
                try:
                    base_name = os.path.basename(image_path)
                    name, ext = os.path.splitext(base_name)
                    mask_path = os.path.join(final_input_dir, f"{name}_mask.png")
                    output_path = os.path.join(inpaint_output_dir, f"{name}_{selected_method}{ext}")

                    if os.path.exists(mask_path):
                        result_image = process_image_with_mask(image_path, mask_path, selected_method)
                        cv2.imwrite(output_path, result_image)
                    else:
                        print(f"Mask not found for {image_path}. Skipping {selected_method}.")
                except Exception as e:
                    print(f"Error applying {selected_method} to {image_path}: {e}")


def run_pipeline():
    root = Tk()
    root.withdraw()

    image_paths = []
    FileHandler.choose_files_or_folder(root, image_paths)

    if image_paths:
        base_output_dir = filedialog.askdirectory(title="Select Folder to Save Results")
        mask_output_dir, inpaint_output_dir, final_input_dir = FileHandler.setup_output_directories(base_output_dir)

        image_paths = FileHandler.convert_images_to_png(image_paths, final_input_dir)

        if len(image_paths) > 1:
            create_masks_choice = []
            PipelineManager.ask_yes_no_question(root, "Do you want to create masks?", create_masks_choice)
            create_masks = create_masks_choice[0].lower()
            if create_masks == 'yes':
                PipelineManager.manage_multiple_images_without_masks(root, image_paths, mask_output_dir,
                                                                     final_input_dir)
            else:
                PipelineManager.manage_multiple_images_with_masks(image_paths, final_input_dir)
        else:
            PipelineManager.manage_single_image(root, image_paths, mask_output_dir, final_input_dir)

        use_inpainting = messagebox.askyesno("Select Method", "Do you want to use inpainting?")

        if use_inpainting:
            PipelineManager.use_inpainting_method(root, final_input_dir, inpaint_output_dir)
        else:
            PipelineManager.use_trivial_anonymization_method(root, image_paths, final_input_dir, inpaint_output_dir)

        print("Pipeline processing complete.")


if __name__ == "__main__":
    run_pipeline()
