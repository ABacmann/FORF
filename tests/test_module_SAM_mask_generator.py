import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2
from framework.mask_generator.SAM_mask_generator import SamSegmentGUI, get_clicked_point

class TestSamSegmentGUI(unittest.TestCase):

    def setUp(self):
        config = MagicMock()
        config.sam_ckpt_path = 'mock_ckpt.pth'
        config.sam_segment_script = 'mock_script.py'
        config.sam_model_type = 'mock_model'
        config.dilate_kernel_size = 5
        config.point_labels = 1

        self.gui = SamSegmentGUI(config)

    @patch('framework.mask_generator.SAM_mask_generator.filedialog.askopenfilename')
    def test_select_image(self, mock_askopenfilename):
        mock_askopenfilename.return_value = 'mock_image.png'
        self.gui.select_image()
        self.assertEqual(self.gui.input_img, 'mock_image.png')

    @patch('framework.mask_generator.SAM_mask_generator.filedialog.askdirectory')
    def test_select_output_directory(self, mock_askdirectory):
        mock_askdirectory.return_value = 'mock_output_dir'
        self.gui.select_output_directory()
        self.assertEqual(self.gui.output_dir, 'mock_output_dir')

    @patch('framework.mask_generator.SAM_mask_generator.get_clicked_point')
    def test_select_points_with_opencv(self, mock_get_clicked_point):
        mock_get_clicked_point.return_value = [(10, 20)]
        self.gui.input_img = 'mock_image.png'
        self.gui.select_points_with_opencv()
        self.assertEqual(self.gui.point_coords, [(10, 20)])

    @patch('framework.mask_generator.SAM_mask_generator.subprocess.run')
    def test_run_sam_segment(self, mock_subprocess_run):
        self.gui.input_img = 'mock_image.png'
        self.gui.output_dir = 'mock_output_dir'
        self.gui.point_coords = [(10, 20)]

        self.gui.run_sam_segment()

        expected_command = [
            "python", 'mock_script.py',
            "--input_img", 'mock_image.png',
            "--point_coords", '(10, 20)',
            "--point_labels", '1',
            "--dilate_kernel_size", '5',
            "--output_dir", 'mock_output_dir',
            "--sam_model_type", 'mock_model',
            "--sam_ckpt", 'mock_ckpt.pth'
        ]

        mock_subprocess_run.assert_called_once_with(expected_command)

    @patch('framework.mask_generator.SAM_mask_generator.filedialog.askopenfilename')
    @patch('framework.mask_generator.SAM_mask_generator.filedialog.askdirectory')
    @patch('framework.mask_generator.SAM_mask_generator.get_clicked_point')
    @patch('framework.mask_generator.SAM_mask_generator.subprocess.run')
    def test_run(self, mock_subprocess_run, mock_get_clicked_point, mock_askdirectory, mock_askopenfilename):
        mock_askopenfilename.return_value = 'mock_image.png'
        mock_askdirectory.return_value = 'mock_output_dir'
        mock_get_clicked_point.return_value = [(10, 20)]

        self.gui.run()

        self.assertEqual(self.gui.input_img, 'mock_image.png')
        self.assertEqual(self.gui.output_dir, 'mock_output_dir')
        self.assertEqual(self.gui.point_coords, [(10, 20)])

        expected_command = [
            "python", 'mock_script.py',
            "--input_img", 'mock_image.png',
            "--point_coords", '(10, 20)',
            "--point_labels", '1',
            "--dilate_kernel_size", '5',
            "--output_dir", 'mock_output_dir',
            "--sam_model_type", 'mock_model',
            "--sam_ckpt", 'mock_ckpt.pth'
        ]

        mock_subprocess_run.assert_called_once_with(expected_command)


class TestGetClickedPoint(unittest.TestCase):

    @patch('cv2.imread')
    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    @patch('cv2.destroyAllWindows')
    @patch('cv2.setMouseCallback')
    @patch('cv2.namedWindow')
    def test_get_clicked_point(self, mock_namedWindow, mock_setMouseCallback, mock_destroyAllWindows, mock_waitKey,
                               mock_imshow, mock_imread):
        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        # Create a generator that yields 1 multiple times and then 27 to simulate ESC key press
        def waitKey_gen():
            for _ in range(100):
                yield 1
            yield 27  # ASCII for ESC key to simulate closing the window

        mock_waitKey.side_effect = waitKey_gen()

        # Container for the coordinates of the clicked points
        clicked_points = []

        # Simulate a left mouse button click at (50, 50) and a right mouse button click to exit
        def mock_setMouseCallback_func(winname, callback):
            # First simulate a left mouse button click at (50, 50)
            callback(cv2.EVENT_LBUTTONDOWN, 50, 50, None, clicked_points)
            # Then simulate a right mouse button click to exit
            callback(cv2.EVENT_RBUTTONDOWN, 50, 50, None, clicked_points)

        mock_setMouseCallback.side_effect = mock_setMouseCallback_func

        point = get_clicked_point('mock_image.png')

        self.assertIsInstance(point, list)
        self.assertEqual(len(point), 2)
        self.assertEqual(point, [50, 50])

if __name__ == '__main__':
    unittest.main()
