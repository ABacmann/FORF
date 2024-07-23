import os
import sys
import unittest
from unittest.mock import patch
from framework.models.Big_Lama_Fourier import BigLamaModel, FineTunedLamaModel, FineTunedLamaModel2, BaseBigLamaModel


class TestBaseBigLamaModel(unittest.TestCase):

    def setUp(self):
        self.model_dir = '/mock/model/dir'
        self.config_path = '/mock/config.yaml'
        self.checkpoint = 'mock.ckpt'
        self.torch_home = '/mock/torch/home'
        self.model = BaseBigLamaModel(self.model_dir, self.config_path, self.checkpoint, self.torch_home)

    @patch('framework.models.Big_Lama_Fourier.sys.path', new_callable=list)
    @patch.dict('framework.models.Big_Lama_Fourier.os.environ', {}, clear=True)
    def test_prepare_environment(self, _):
        # Call the method to set the environment
        self.model.prepare_environment()

        # Expected paths
        expected_forf_path = os.path.abspath(os.path.join(self.torch_home, '..'))
        expected_lama_path = os.path.join(expected_forf_path, 'lama')
        expected_python_path = os.pathsep.join(
            [expected_forf_path, expected_lama_path, os.path.join(expected_lama_path, 'bin')])

        # Debug prints
        print(f"Expected FORF Path: {expected_forf_path}")
        print(f"Expected LAMA Path: {expected_lama_path}")
        print(f"Expected PYTHONPATH: {expected_python_path}")
        print(f"Actual TORCH_HOME: {os.environ.get('TORCH_HOME')}")
        print(f"Actual PYTHONPATH: {os.environ.get('PYTHONPATH')}")
        print(f"sys.path: {sys.path}")

        # Verify environment variables are set correctly
        self.assertEqual(os.environ['TORCH_HOME'], self.torch_home)
        self.assertEqual(os.environ['PYTHONPATH'], expected_python_path)

        # Verify sys.path is updated correctly
        self.assertIn(expected_forf_path, sys.path)
        self.assertIn(expected_lama_path, sys.path)
        self.assertIn(os.path.join(expected_lama_path, 'bin'), sys.path)

    def test_set_directories(self):
        indir = '/mock/input/dir'
        outdir = '/mock/output/dir'
        self.model.set_directories(indir, outdir)
        self.assertEqual(self.model.indir, indir)
        self.assertEqual(self.model.outdir, outdir)

    @patch('framework.models.Big_Lama_Fourier.subprocess.run')
    def test_run_prediction(self, mock_subprocess_run):
        indir = '/mock/input/dir'
        outdir = '/mock/output/dir'
        self.model.set_directories(indir, outdir)
        self.model.run_prediction()

        expected_command = [
            sys.executable, os.path.join(self.torch_home, 'bin', 'predict.py'),
            f'model.path={self.model_dir}',
            f'model.checkpoint={self.checkpoint}',
            f'indir={indir}',
            f'outdir={outdir}'
        ]

        mock_subprocess_run.assert_called_once_with(expected_command, check=True)


class TestBigLamaModel(unittest.TestCase):

    def test_initialization(self):
        model = BigLamaModel(torch_home='/mock/torch/home')
        self.assertEqual(model.model_dir, r'C:\Users\Administrator\Projects\FORF\lama\models\big-lama')
        self.assertEqual(model.config_path,
                         os.path.join(r'C:\Users\Administrator\Projects\FORF\lama\models\big-lama', 'config.yaml'))
        self.assertEqual(model.checkpoint, 'best.ckpt')
        self.assertEqual(model.torch_home, '/mock/torch/home')


class TestFineTunedLamaModel(unittest.TestCase):

    def test_initialization(self):
        model = FineTunedLamaModel(torch_home='/mock/torch/home')
        self.assertEqual(model.model_dir, r'C:\Users\Administrator\Projects\FORF\lama\models\fine-tuned-big-lama')
        self.assertEqual(model.config_path,
                         os.path.join(r'C:\Users\Administrator\Projects\FORF\lama\models\fine-tuned-big-lama',
                                      'config.yaml'))
        self.assertEqual(model.checkpoint, 'fine-tuned-epoch12.ckpt')
        self.assertEqual(model.torch_home, '/mock/torch/home')


class TestFineTunedLamaModel2(unittest.TestCase):

    def test_initialization(self):
        model = FineTunedLamaModel2(torch_home='/mock/torch/home')
        self.assertEqual(model.model_dir, r'C:\Users\Administrator\Projects\FORF\lama\models\fine-tuned2-big-lama')
        self.assertEqual(model.config_path,
                         os.path.join(r'C:\Users\Administrator\Projects\FORF\lama\models\fine-tuned2-big-lama',
                                      'config.yaml'))
        self.assertEqual(model.checkpoint, 'fine-tuned2-epoch13.ckpt')
        self.assertEqual(model.torch_home, '/mock/torch/home')


if __name__ == '__main__':
    unittest.main()
