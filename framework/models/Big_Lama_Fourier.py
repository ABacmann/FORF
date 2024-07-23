import os
import subprocess
import sys


class BaseBigLamaModel:
    def __init__(self, model_dir, config_path, checkpoint, torch_home=None):
        self.model_dir = model_dir
        self.config_path = config_path
        self.checkpoint = checkpoint
        self.torch_home = torch_home if torch_home else os.getcwd()
        self.indir = None
        self.outdir = None

    def prepare_environment(self):
        forf_path = os.path.abspath(os.path.join(self.torch_home, '..'))  # Set FORF as the base directory
        os.environ['TORCH_HOME'] = self.torch_home
        lama_path = os.path.join(forf_path, 'lama')
        os.environ['PYTHONPATH'] = os.pathsep.join([forf_path, lama_path, os.path.join(lama_path, 'bin')])
        sys.path.extend([forf_path, lama_path, os.path.join(lama_path, 'bin')])

    def set_directories(self, indir, outdir):
        self.indir = indir
        self.outdir = outdir

    def run_prediction(self):
        if not self.indir or not self.outdir:
            raise ValueError("Input and output directories must be set.")

        predict_command = [
            sys.executable, os.path.join(self.torch_home, 'bin', 'predict.py'),
            #f'refine={True}', # Set True to activate the Refinement feature from Kulshreshtha et al.
            f'model.path={self.model_dir}',
            f'model.checkpoint={self.checkpoint}',
            f'indir={self.indir}',
            f'outdir={self.outdir}'
        ]

        subprocess.run(predict_command, check=True)


class BigLamaModel(BaseBigLamaModel):
    def __init__(self, torch_home=None):
        model_dir = r'C:\Users\Administrator\Projects\FORF\lama\models\big-lama'
        config_path = os.path.join(model_dir, 'config.yaml')
        checkpoint = 'best.ckpt'
        super().__init__(model_dir, config_path, checkpoint, torch_home)


class FineTunedLamaModel(BaseBigLamaModel):
    def __init__(self, torch_home=None):
        model_dir = r'C:\Users\Administrator\Projects\FORF\lama\models\fine-tuned-big-lama'
        config_path = os.path.join(model_dir, 'config.yaml')
        checkpoint = 'fine-tuned-epoch12.ckpt'
        super().__init__(model_dir, config_path, checkpoint, torch_home)


class FineTunedLamaModel2(BaseBigLamaModel):
    def __init__(self, torch_home=None):
        model_dir = r'C:\Users\Administrator\Projects\FORF\lama\models\fine-tuned2-big-lama'
        config_path = os.path.join(model_dir, 'config.yaml')
        checkpoint = 'fine-tuned2-epoch13.ckpt'
        super().__init__(model_dir, config_path, checkpoint, torch_home)


if __name__ == "__main__":
    # Choose the model you want to use
    model_type = 'fine_tuned'  # or 'fine_tuned'

    if model_type == 'big':
        model = BigLamaModel(torch_home=r'C:\Users\Administrator\Projects\FORF\lama')
    elif model_type == 'fine_tuned':
        model = FineTunedLamaModel(torch_home=r'C:\Users\Administrator\Projects\FORF\lama')
    else:
        raise ValueError("Invalid model type specified")

    # Prepare the environment
    model.prepare_environment()

    # Set input and output directories
    model.set_directories(
        indir=r'C:\Users\Administrator\Projects\FORF\Data\LaMa_Medical_Images',
        outdir=r'C:\Users\Administrator\Projects\FORF\Data\Results\Inpainted_Images'
    )

    # Run prediction
    model.run_prediction()
