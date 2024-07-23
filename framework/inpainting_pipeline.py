from PIL import Image
import os


class InpaintingManager:
    def __init__(self, inpainting_model):
        self.inpainting_model = inpainting_model

    def process_image(self, image_path, mask_path, output_path):
        """
        Process a single image and apply inpainting using the provided model.

        Args:
        - image_path (str): The path to the input image.
        - mask_path (str): The path to the mask image.
        - output_path (str): The path to save the inpainted image.

        Returns:
        None
        """
        try:
            # Load the image and mask
            image = Image.open(image_path)
            mask = Image.open(mask_path).convert('L')  # Ensure the mask is in grayscale

            # Apply inpainting
            inpainted_image = self.inpainting_model.inpaint(image, mask)

            # Save the inpainted image
            inpainted_image.save(output_path)
            print(f"Inpainted image saved at {output_path}")

        except Exception as e:
            print(f"Error processing {image_path} with mask {mask_path}: {e}")

    def process_directory(self, input_dir, output_dir):
        """
        Process all images in the input directory and apply inpainting.

        Args:
        - input_dir (str): The path to the input directory containing images and masks.
        - output_dir (str): The path to save the inpainted images.

        Returns:
        None
        """
        print("Starting directory processing...")

        # If the model is directory-based, set directories directly
        if hasattr(self.inpainting_model, 'set_directories'):
            print("Detected a directory-based model. Setting directories and running prediction.")
            self.inpainting_model.set_directories(indir=input_dir, outdir=output_dir)
            self.inpainting_model.run_prediction()
        else:
            print("Detected an image-based model. Processing each image individually.")
            # Process each image individually
            for file_name in os.listdir(input_dir):
                if file_name.endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(input_dir, file_name)
                    base_name, ext = os.path.splitext(file_name)
                    mask_path = os.path.join(input_dir, f"{base_name}_mask.png")
                    output_path = os.path.join(output_dir, f"{base_name}_inpainted{ext}")

                    if os.path.exists(mask_path):
                        print(f"Processing {image_path} with {mask_path}")
                        self.process_image(image_path, mask_path, output_path)
                    else:
                        print(f"Mask not found for {image_path}. Skipping inpainting.")
