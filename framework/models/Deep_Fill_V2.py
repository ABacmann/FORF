import os
import subprocess
import sys
from pathlib import Path


class DeepFillV2:
    def __init__(self, torch_home=None):
        self.model_path = r'C:\Users\Administrator\Projects\FORF\deepfillv2_pytorch\pretrained\states_pt_places2.pth'
        self.torch_home = torch_home if torch_home else os.getcwd()
        self.indir = None
        self.outdir = None

    def prepare_environment(self):
        forf_path = Path(self.torch_home).parent.resolve()
        os.environ['TORCH_HOME'] = str(self.torch_home)
        deepfill_path = forf_path / 'deepfillv2_pytorch'
        os.environ['PYTHONPATH'] = os.pathsep.join([str(forf_path), str(deepfill_path)])
        sys.path.extend([str(forf_path), str(deepfill_path)])

    def set_directories(self, indir, outdir):
        self.indir = indir
        self.outdir = outdir

    def run_prediction(self):
        if not self.indir or not self.outdir:
            raise ValueError("Input and output directories must be set.")

        if not os.path.isdir(self.indir):
            raise NotADirectoryError(f"The provided input path is not a directory: {self.indir}")

        images = [f for f in os.listdir(self.indir) if f.endswith(('jpg', 'jpeg', 'png'))]

        if not images:
            raise FileNotFoundError(f"No image files found in the directory: {self.indir}")

        for image in images:
            image_path = os.path.join(self.indir, image)
            mask_path = os.path.join(self.indir, f"{Path(image).stem}_mask{Path(image).suffix}")
            output_path = os.path.join(self.outdir, f"{Path(image).stem}_out{Path(image).suffix}")

            if not os.path.exists(mask_path):
                print(f"Mask file for {image} not found, skipping.")
                continue

            predict_command = [
                sys.executable, os.path.join(self.torch_home, 'deepfillv2_pytorch', 'test.py'),
                '--image', image_path,
                '--mask', mask_path,
                '--out', output_path,
                '--checkpoint', self.model_path
            ]

            try:
                subprocess.run(predict_command, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Prediction failed for {image}: {e}")


# Usage example
if __name__ == '__main__':
    model = DeepFillV2(torch_home=r'C:\Users\Administrator\Projects\FORF')
    model.prepare_environment()
    model.set_directories(
        indir=r'C:\Users\Administrator\Projects\FORF\framework\data\Tess',
        outdir=r'C:\Users\Administrator\Projects\FORF\framework\data\Results')
    model.run_prediction()
