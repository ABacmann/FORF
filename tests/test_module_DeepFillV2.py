import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from framework.models.Deep_Fill_V2 import DeepFillV2
from pathlib import Path

class TestDeepFillV2(unittest.TestCase):

    def setUp(self):
        self.torch_home = '/mock/torch/home'
        self.model = DeepFillV2(self.torch_home)
        self.indir = '/mock/input/dir'
        self.outdir = '/mock/output/dir'

    @patch('framework.models.Deep_Fill_V2.sys.path', new_callable=list)
    @patch.dict('framework.models.Deep_Fill_V2.os.environ', {}, clear=True)
    def test_prepare_environment(self, _):
        # Call the method to set the environment
        self.model.prepare_environment()

        # Expected paths
        expected_forf_path = Path(self.torch_home).parent.resolve()
        expected_deepfill_path = expected_forf_path / 'deepfillv2_pytorch'
        expected_python_path = os.pathsep.join([str(expected_forf_path), str(expected_deepfill_path)])

        # Debug prints
        print(f"Expected FORF Path: {expected_forf_path}")
        print(f"Expected DeepFill Path: {expected_deepfill_path}")
        print(f"Expected PYTHONPATH: {expected_python_path}")
        print(f"Actual TORCH_HOME: {os.environ.get('TORCH_HOME')}")
        print(f"Actual PYTHONPATH: {os.environ.get('PYTHONPATH')}")
        print(f"sys.path: {sys.path}")

        # Verify environment variables are set correctly
        self.assertEqual(os.environ['TORCH_HOME'], self.torch_home)
        self.assertEqual(os.environ['PYTHONPATH'], expected_python_path)

        # Verify sys.path is updated correctly
        self.assertIn(str(expected_forf_path), sys.path)
        self.assertIn(str(expected_deepfill_path), sys.path)

    def test_set_directories(self):
        self.model.set_directories(self.indir, self.outdir)
        self.assertEqual(self.model.indir, self.indir)
        self.assertEqual(self.model.outdir, self.outdir)

    @patch('framework.models.Deep_Fill_V2.subprocess.run')
    @patch('framework.models.Deep_Fill_V2.os.listdir')
    @patch('framework.models.Deep_Fill_V2.os.path.isdir')
    @patch('framework.models.Deep_Fill_V2.os.path.exists')
    def test_run_prediction(self, mock_exists, mock_isdir, mock_listdir, mock_subprocess_run):
        # Set up mocks
        mock_isdir.return_value = True
        mock_listdir.return_value = ['image1.jpg', 'image2.png']
        mock_exists.side_effect = lambda x: x.endswith('_mask.jpg') or x.endswith('_mask.png')

        self.model.set_directories(self.indir, self.outdir)
        self.model.run_prediction()

        expected_commands = [
            [
                sys.executable, os.path.join(self.torch_home, 'deepfillv2_pytorch', 'test.py'),
                '--image', os.path.join(self.indir, 'image1.jpg'),
                '--mask', os.path.join(self.indir, 'image1_mask.jpg'),
                '--out', os.path.join(self.outdir, 'image1_out.jpg'),
                '--checkpoint', self.model.model_path
            ],
            [
                sys.executable, os.path.join(self.torch_home, 'deepfillv2_pytorch', 'test.py'),
                '--image', os.path.join(self.indir, 'image2.png'),
                '--mask', os.path.join(self.indir, 'image2_mask.png'),
                '--out', os.path.join(self.outdir, 'image2_out.png'),
                '--checkpoint', self.model.model_path
            ]
        ]

        self.assertEqual(mock_subprocess_run.call_count, 2)
        mock_subprocess_run.assert_any_call(expected_commands[0], check=True)
        mock_subprocess_run.assert_any_call(expected_commands[1], check=True)

    @patch('framework.models.Deep_Fill_V2.os.listdir')
    @patch('framework.models.Deep_Fill_V2.os.path.isdir')
    def test_run_prediction_no_images(self, mock_isdir, mock_listdir):
        mock_isdir.return_value = True
        mock_listdir.return_value = []

        self.model.set_directories(self.indir, self.outdir)
        with self.assertRaises(FileNotFoundError):
            self.model.run_prediction()

    @patch('framework.models.Deep_Fill_V2.os.path.isdir')
    def test_run_prediction_invalid_input_directory(self, mock_isdir):
        mock_isdir.return_value = False

        self.model.set_directories(self.indir, self.outdir)
        with self.assertRaises(NotADirectoryError):
            self.model.run_prediction()


if __name__ == '__main__':
    unittest.main()
