import os
import shutil
from tkinter import filedialog, simpledialog, Tk, messagebox, Toplevel, Label, Button, Frame
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
        output_dir = os.path.join(base_dir, sub_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        else:
            print(f"Directory already exists: {output_dir}")
        return output_dir

    @staticmethod
    def setup_output_directories(base_output_dir):
        mask_output_dir = FileHandler.create_output_directory(base_output_dir, "Masks")
        inpaint_output_dir = FileHandler.create_output_directory(base_output_dir, "Inpainted_Images")
        final_input_dir = FileHandler.create_output_directory(base_output_dir, "Ground_Truth")
        return mask_output_dir, inpaint_output_dir, final_input_dir

    @staticmethod
    def choose_files_or_folder(root, image_paths):
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
        png_paths = []
        for image_path in image_paths:
            base_name = os.path.basename(image_path)
            name, _ = os.path.splitext(base_name)
            png_path = os.path.join(final_input_dir, f"{name}.png")

            if image_path.lower().endswith('.dcm'):
                dicom_to_png(image_path, png_path)
                png_paths.append(png_path)
            elif image_path.lower().endswith(('.jpg', '.jpeg')):
                img = Image.open(image_path)
                img.save(png_path)
                png_paths.append(png_path)
            else:
                png_paths.append(image_path)

        return png_paths


class MaskGeneratorHandler:
    @staticmethod
    def generate_manual_mask(image_path, mask_output_dir):
        """
        Generates a manual mask for the given image by opening a MaskGenerator GUI.
        Waits for the GUI window to close before returning the path to the generated mask.

        Parameters:
        - image_path (str): The path to the image for which the mask is to be generated.
        - mask_output_dir (str): The directory where the generated mask will be saved.

        Returns:
        - str: The path to the generated mask file.
        """
        mask_gen = MaskGenerator(image_path, mask_output_dir)
        mask_gen.root.wait_window(mask_gen.root)
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        return os.path.join(mask_output_dir, f"{name}_mask.png")

    @staticmethod
    def generate_sam_mask(image_path, mask_output_dir):
        """
        Generates a SAM (Segment Anything Model) mask for the given image using the SamSegmentGUI.
        Executes the SAM segmentation process and returns the paths to the generated masks and their visual representations.

        Parameters:
        - image_path (str): The path to the image for which the mask is to be generated.
        - mask_output_dir (str): The directory where the generated masks will be saved.

        Returns:
        - tuple: A tuple containing two lists:
          1. Paths to the generated mask files.
          2. Paths to the visual representations of the masks.
        """
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

    @staticmethod
    def copy_generated_mask(generated_mask_path, final_input_dir, new_name):
        """
        Copies a generated mask to a new location with a new name.
        Prints a confirmation message if the mask is copied successfully, or an error message if the mask is not found.

        Parameters:
        - generated_mask_path (str): The path to the generated mask to be copied.
        - final_input_dir (str): The directory where the mask will be copied to.
        - new_name (str): The new name for the copied mask file.

        Returns:
        - None
        """
        if os.path.exists(generated_mask_path):
            destination = os.path.join(final_input_dir, new_name)
            shutil.copy(generated_mask_path, destination)
            print(f"Copied mask to: {destination}")
        else:
            print(f"Mask not found: {generated_mask_path}")

    @staticmethod
    def choose_mask(visual_masks):
        """
        Opens a selection window allowing the user to choose one of the visual masks.
        Displays each visual mask in a grid with a button to select it.
        Returns the index of the chosen mask.

        Parameters:
        - visual_masks (list): A list of paths to the visual representations of the masks.

        Returns:
        - int: The index of the chosen mask.
        """
        root = Toplevel()
        root.title("Choose a Mask")
        chosen_mask = [None]

        def select_mask(mask_index):
            chosen_mask[0] = mask_index
            root.quit()

        for i, visual_mask in enumerate(visual_masks):
            img = Image.open(visual_mask)
            img = img.resize((300, 300))
            img = ImageTk.PhotoImage(img)
            panel = Label(root, image=img)
            panel.image = img
            panel.grid(row=0, column=i)
            button = Button(root, text=f"Select Mask {i}", command=lambda i=i: select_mask(i))
            button.grid(row=1, column=i)

        root.mainloop()
        root.destroy()
        return chosen_mask[0]


class InpaintingHandler:
    @staticmethod
    def process_single_image(image_path, mask_output_dir, final_input_dir, mask_method):
        """
        Processes a single image for inpainting by generating a mask and copying the image and mask to the final input directory.
        Depending on the mask_method, it either generates a manual mask or uses the SAM mask generation method.

        Parameters:
        - image_path (str): The path to the image to be processed.
        - mask_output_dir (str): The directory where the generated masks will be saved.
        - final_input_dir (str): The directory where the final input images and masks will be copied.
        - mask_method (str): The method for generating the mask ('manual' or 'sam').

        Returns:
        - None
        """
        print(f"Processing single image: {image_path}")

        # Ensure the source and destination are not the same file
        destination_path = Path(final_input_dir) / Path(image_path).name
        if Path(image_path).resolve() == destination_path.resolve():
            print(f"Source and destination are the same file: {image_path}. Skipping copy.")
        else:
            shutil.copy(image_path, final_input_dir)
            print(f"Copied image to final input dir: {final_input_dir}")

        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        if mask_method == 'manual':
            generated_mask_path = MaskGeneratorHandler.generate_manual_mask(image_path, mask_output_dir)
            print(f"Generated manual mask: {generated_mask_path}")
            MaskGeneratorHandler.copy_generated_mask(generated_mask_path, final_input_dir, f"{name}_mask.png")
        else:
            generated_masks, visual_masks = MaskGeneratorHandler.generate_sam_mask(image_path, mask_output_dir)
            print(f"Generated SAM masks: {generated_masks}")
            chosen_mask_index = MaskGeneratorHandler.choose_mask(visual_masks)
            chosen_mask = generated_masks[chosen_mask_index]
            print(f"Chosen mask: {chosen_mask}")
            MaskGeneratorHandler.copy_generated_mask(chosen_mask, final_input_dir, f"{name}_mask.png")

    @staticmethod
    def process_multiple_images(image_paths, mask_output_dir, final_input_dir, mask_method):
        """
        Processes multiple images for inpainting by generating masks and copying the images and masks to the final input directory.
        Depending on the mask_method, it either generates manual masks or uses the SAM mask generation method for each image.

        Parameters:
        - image_paths (list): A list of paths to the images to be processed.
        - mask_output_dir (str): The directory where the generated masks will be saved.
        - final_input_dir (str): The directory where the final input images and masks will be copied.
        - mask_method (str): The method for generating the masks ('manual' or 'sam').

        Returns:
        - None
        """
        for image_path in image_paths:
            shutil.copy(image_path, final_input_dir)

        for image_path in image_paths:
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            if mask_method == 'manual':
                generated_mask_path = MaskGeneratorHandler.generate_manual_mask(image_path, mask_output_dir)
                MaskGeneratorHandler.copy_generated_mask(generated_mask_path, final_input_dir, f"{name}_mask.png")
            else:
                generated_masks, visual_masks = MaskGeneratorHandler.generate_sam_mask(image_path, mask_output_dir)
                chosen_mask_index = MaskGeneratorHandler.choose_mask(visual_masks)
                chosen_mask = generated_masks[chosen_mask_index]
                MaskGeneratorHandler.copy_generated_mask(chosen_mask, final_input_dir, f"{name}_mask.png")


class PipelineManager:
    @staticmethod
    def ask_yes_no_question(root, question, result):
        """
        Asks a yes/no question using a messagebox and appends the result to the provided list.

        Parameters:
        - root (Tk): The root window of the Tkinter application.
        - question (str): The question to be asked.
        - result (list): A list to which the answer ('yes' or 'no') will be appended.

        Returns:
        - None
        """
        choice = messagebox.askquestion("Question", question)
        result.append('yes' if choice == 'yes' else 'no')

    @staticmethod
    def select_mask_generation_method(root):
        """
        Opens a selection window for the user to choose a mask generation method.
        The methods available are 'manual' and 'SAM'.

        Parameters:
        - root (Tk): The root window of the Tkinter application.

        Returns:
        - str: The selected mask generation method.
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
        Opens a selection window for the user to choose an inpainting model.
        The available models are listed in the models array.

        Parameters:
        - root (Tk): The root window of the Tkinter application.

        Returns:
        - str: The selected inpainting model.
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

        Parameters:
        - selected_model_name (str): The name of the selected inpainting model.

        Returns:
        - object: An instance of the selected inpainting model, or None if the model is unknown.
        """
        if selected_model_name == 'DeepFillV2':
            return DeepFillV2()
        elif selected_model_name == 'BigLamaModel':
            return BigLamaModel(torch_home=os.getenv('TORCH_HOME'))
        elif selected_model_name == 'FineTunedLamaModel':
            return FineTunedLamaModel(torch_home=os.getenv('TORCH_HOME'))
        elif selected_model_name == 'FineTunedLamaModel2':
            return FineTunedLamaModel2(torch_home=os.getenv('TORCH_HOME'))
        else:
            print(f"Unknown model selected: {selected_model_name}")
            return None
    @staticmethod
    def select_removal_method(root):
        """
        Opens a selection window for the user to choose a trivial anonymization method.
        The methods available are 'blur', 'pixelate', 'mask', 'distort', and 'inpaint'.

        Parameters:
        - root (Tk): The root window of the Tkinter application.

        Returns:
        - str: The selected removal method.
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
    def manage_multiple_images_with_masks(root, image_paths, mask_output_dir, final_input_dir):
        """
        Manages the processing of multiple images with masks by selecting a mask generation method
        and processing each image accordingly.

        Parameters:
        - root (Tk): The root window of the Tkinter application.
        - image_paths (list): A list of paths to the images to be processed.
        - mask_output_dir (str): The directory where the generated masks will be saved.
        - final_input_dir (str): The directory where the final input images and masks will be copied.

        Returns:
        - None
        """
        mask_method = PipelineManager.select_mask_generation_method(root)
        if not mask_method:
            print("No mask generation method selected. Exiting.")
            return
        InpaintingHandler.process_multiple_images(image_paths, mask_output_dir, final_input_dir, mask_method)

    @staticmethod
    def manage_multiple_images_without_masks(image_paths, final_input_dir):
        """
        Manages the processing of multiple images without generating masks.
        Copies the images and their corresponding masks (if they exist) to the final input directory.

        Parameters:
        - image_paths (list): A list of paths to the images to be processed.
        - final_input_dir (str): The directory where the final input images and masks will be copied.

        Returns:
        - None
        """
        for image_path in image_paths:
            shutil.copy(image_path, final_input_dir)
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            mask_path = os.path.join(os.path.dirname(image_path), f"{name}_mask.png")
            if os.path.exists(mask_path):
                shutil.copy(mask_path, final_input_dir)
            else:
                print(f"No mask found for {image_path}")

    @staticmethod
    def manage_single_image(root, image_paths, mask_output_dir, final_input_dir):
        """
        Manages the processing of a single image by selecting a mask generation method
        and processing the image accordingly.

        Parameters:
        - root (Tk): The root window of the Tkinter application.
        - image_paths (list): A list containing the path to the image to be processed.
        - mask_output_dir (str): The directory where the generated mask will be saved.
        - final_input_dir (str): The directory where the final input image and mask will be copied.

        Returns:
        - None
        """
        mask_method = PipelineManager.select_mask_generation_method(root)
        if not mask_method:
            print("No mask generation method selected. Exiting.")
            return
        InpaintingHandler.process_single_image(image_paths[0], mask_output_dir, final_input_dir, mask_method)

    @staticmethod
    def use_inpainting_method(root, final_input_dir, inpaint_output_dir):
        """
        Uses an inpainting method to process images in the final input directory and save the output
        to the inpaint output directory.

        Parameters:
        - root (Tk): The root window of the Tkinter application.
        - final_input_dir (str): The directory containing the final input images and masks.
        - inpaint_output_dir (str): The directory where the inpainted images will be saved.

        Returns:
        - None
        """
        selected_model_name = PipelineManager.select_model_manually(root)
        inpainting_model = PipelineManager.select_inpainting_model(selected_model_name)
        if inpainting_model is None:
            return
        pipeline = InpaintingManager(inpainting_model)
        pipeline.process_directory(final_input_dir, inpaint_output_dir)

    @staticmethod
    def use_trivial_anonymization_method(root, image_paths, final_input_dir, inpaint_output_dir):
        """
        Uses a trivial anonymization method to process images based on the selected method.
        The available methods are 'blur', 'pixelate', 'mask', 'distort', and 'inpaint'.

        Parameters:
        - root (Tk): The root window of the Tkinter application.
        - image_paths (list): A list of paths to the images to be processed.
        - final_input_dir (str): The directory containing the final input images and masks.
        - inpaint_output_dir (str): The directory where the processed images will be saved.

        Returns:
        - None
        """
        selected_method = PipelineManager.select_removal_method(root)
        if selected_method not in ['blur', 'pixelate', 'mask', 'distort', 'inpaint']:
            print(f"Unknown method selected: {selected_method}")
            return

        for image_path in image_paths:
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            mask_path = os.path.join(final_input_dir, f"{name}_mask.png")
            output_path = os.path.join(inpaint_output_dir, f"{name}_{selected_method}{ext}")

            if os.path.exists(mask_path):
                result_image = process_image_with_mask(image_path, mask_path, selected_method)
                cv2.imwrite(output_path, result_image)
            else:
                print(f"Mask not found for {image_path}. Skipping {selected_method}.")


def run_pipeline():
    root = Tk()
    root.withdraw()

    # Select files or a folder of images
    image_paths = []
    FileHandler.choose_files_or_folder(root, image_paths)

    if image_paths:
        # Set up output directories
        base_output_dir = filedialog.askdirectory(title="Select Folder to Save Results")
        mask_output_dir, inpaint_output_dir, final_input_dir = FileHandler.setup_output_directories(base_output_dir)

        # Convert images (DICOM, JPEG, JPG) to PNG
        image_paths = FileHandler.convert_images_to_png(image_paths, final_input_dir)

        # Determine if masks need to be created
        if len(image_paths) > 1:  # Multiple images selected
            create_masks_choice = []
            PipelineManager.ask_yes_no_question(root, "Do you want to create masks?", create_masks_choice)
            create_masks = create_masks_choice[0].lower()
            if create_masks == 'yes':
                PipelineManager.manage_multiple_images_with_masks(root, image_paths, mask_output_dir, final_input_dir)
            else:
                PipelineManager.manage_multiple_images_without_masks(image_paths, final_input_dir)
        else:  # Single image selected
            PipelineManager.manage_single_image(root, image_paths, mask_output_dir, final_input_dir)

        # Choose between inpainting and trivial anonymization methods
        use_inpainting = messagebox.askyesno("Select Method", "Do you want to use inpainting?")

        if use_inpainting:
            PipelineManager.use_inpainting_method(root, final_input_dir, inpaint_output_dir)
        else:
            PipelineManager.use_trivial_anonymization_method(root, image_paths, final_input_dir, inpaint_output_dir)

        print("Pipeline processing complete.")


if __name__ == "__main__":
    run_pipeline()

'''        
if use_recommendation == 'yes':
            model_selector = InpaintingModelSelector()

            for image_path in image_paths:
                base_name = os.path.basename(image_path)
                name, ext = os.path.splitext(base_name)
                mask_path = os.path.join(final_input_dir, f"{name}_mask.png")
                output_path = os.path.join(inpaint_output_dir, f"{name}_inpainted{ext}")

                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    recommended_model_name = model_selector.select_model(mask)

                    if recommended_model_name == 'SimpleLamaInpainting':
                        inpainting_model = SimpleLamaInpainting()
                    elif recommended_model_name == 'DeepFillV2':
                        inpainting_model = DeepFillV2()
                    elif recommended_model_name == 'BigLamaModel':
                        inpainting_model = BigLamaModel(torch_home=r'C:a')
                    elif recommended_model_name == 'FineTunedLamaModel':
                        inpainting_model = FineTunedLamaModel(torch_home=r')
                    else:
                        print(f"Unknown model recommended: {recommended_model_name}")
                        continue

                    pipeline = InpaintingPipeline(inpainting_model)
                    pipeline.process_directory(final_input_dir, inpaint_output_dir)
                else:
                    print(f"Mask not found for {image_path}. Skipping inpainting.")

        else:
            # Use manual model selection
            selected_model_name = select_model_manually(root)

            if selected_model_name == 'SimpleLamaInpainting':
                inpainting_model = SimpleLamaInpainting()
            elif selected_model_name == 'DeepFillV2':
                inpainting_model = DeepFillV2()
            elif selected_model_name == 'BigLamaModel':
                inpainting_model = BigLamaModel(torch_home=r'')
            elif selected_model_name == 'FineTunedLamaModel':
                inpainting_model = FineTunedLamaModel(torch_home=r'C:')
            else:
                print(f"Unknown model selected: {selected_model_name}")
                return

            pipeline = InpaintingPipeline(inpainting_model)
            pipeline.process_directory(final_input_dir, inpaint_output_dir)
'''

'''
# Main Pipeline Execution
import os

from tkinter import filedialog, simpledialog, Tk, Label, Button

from framework.mask_generator.Mask_Generator import MaskGenerator
from framework.InpaintingPipeline import InpaintingPipeline
from framework.models.big_laMa import SimpleLamaInpainting
from framework.models.DeepFillV2 import DeepFillV2


def create_output_directory(base_dir, sub_dir):
    output_dir = os.path.join(base_dir, sub_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    else:
        print(f"Directory already exists: {output_dir}")

    return output_dir


def choose_files_or_folder(root, image_paths):
    def select_files():
        selected_files = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")])
        image_paths.extend(selected_files)
        root.quit()  # Close the window

    def select_folder():
        folder_path = filedialog.askdirectory(title="Select Folder with Images")
        if folder_path:
            for file_name in os.listdir(folder_path):
                if file_name.endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(folder_path, file_name))
        root.quit()  # Close the window

    root.deiconify()  # Show the root window
    Label(root, text="Choose Input Type").pack()
    Button(root, text="Select Files", command=select_files).pack()
    Button(root, text="Select Folder", command=select_folder).pack()
    root.mainloop()  # Wait for user interaction


def ask_yes_no_question(root, question, result_container):
    def answer_yes():
        result_container.append('yes')
        root.quit()  # Close the window

    def answer_no():
        result_container.append('no')
        root.quit()  # Close the window

    root.deiconify()  # Show the root window
    Label(root, text=question).pack()
    Button(root, text="Yes", command=answer_yes).pack()
    Button(root, text="No", command=answer_no).pack()
    root.mainloop()  # Wait for user interaction


def run_pipeline():
    root = Tk()
    root.withdraw()  # Hide the root window initially

    image_paths = []
    choose_files_or_folder(root, image_paths)

    if image_paths:
        create_masks_choice = []
        ask_yes_no_question(root, "Do you want to create masks manually?", create_masks_choice)
        create_masks = create_masks_choice[0].lower()

        base_output_dir = filedialog.askdirectory(title="Select Folder to Save Results")
        mask_output_dir = create_output_directory(base_output_dir, "Masks")
        inpaint_output_dir = create_output_directory(base_output_dir, "Inpainted_Images")

        if create_masks == 'yes':
            for image_path in image_paths:
                # Each MaskGenerator runs its own Toplevel window
                mask_gen = MaskGenerator(image_path, mask_output_dir)
                mask_gen.root.wait_window(mask_gen.root)  # Wait until the mask creation window is closed

            # Ask the user to select an inpainting model
        model_choice = simpledialog.askstring("Model Selection",
                                              "Enter 'lama' for SimpleLamaInpainting or 'deepfill' for DeepFillV2:")
        if model_choice == 'lama':
            inpainting_model = SimpleLamaInpainting()
        elif model_choice == 'deepfill':
            inpainting_model = DeepFillV2()
        else:
            print("Invalid model selection.")
            return

        pipeline = InpaintingPipeline(inpainting_model)

        for image_path in image_paths:
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            mask_path = os.path.join(mask_output_dir, f"{name}_mask.png")
            output_path = os.path.join(inpaint_output_dir, f"{name}_inpainted{ext}")

            if os.path.exists(mask_path):
                pipeline.process(image_path, mask_path, output_path)
            else:
                print(f"Mask not found for {image_path}. Skipping inpainting.")

    print("Pipeline processing complete.")


# Run the pipeline
if __name__ == "__main__":
    run_pipeline()
'''
