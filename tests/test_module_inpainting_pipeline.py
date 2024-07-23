import os
import unittest
from unittest.mock import MagicMock, patch, call
from PIL import Image
from framework.inpainting_pipeline import InpaintingManager


class TestInpaintingPipeline(unittest.TestCase):

    def setUp(self):
        self.inpainting_model = MagicMock()
        self.pipeline = InpaintingManager(self.inpainting_model)
        self.image_path = 'mock_image.png'
        self.mask_path = 'mock_image_mask.png'
        self.output_path = 'mock_output.png'
        self.input_dir = '/mock/input/dir'
        self.output_dir = '/mock/output/dir'

    @patch('framework.inpainting_pipeline.Image.open')
    def test_process_image(self, mock_open):
        mock_image = MagicMock(spec=Image.Image)
        mock_mask = MagicMock(spec=Image.Image)
        mock_inpainted_image = MagicMock(spec=Image.Image)

        mock_open.side_effect = [mock_image, mock_mask]
        self.inpainting_model.inpaint.return_value = mock_inpainted_image

        self.pipeline.process_image(self.image_path, self.mask_path, self.output_path)

        mock_open.assert_any_call(self.image_path)
        mock_open.assert_any_call(self.mask_path)
        self.inpainting_model.inpaint.assert_called_once_with(mock_image, mock_mask.convert('L'))
        mock_inpainted_image.save.assert_called_once_with(self.output_path)

    @patch('framework.inpainting_pipeline.os.path.exists')
    @patch('framework.inpainting_pipeline.os.listdir')
    def test_process_directory_with_individual_images(self, mock_listdir, mock_exists):
        mock_listdir.return_value = ['image1.png', 'image2.jpg']
        mock_exists.side_effect = lambda x: x.endswith('_mask.png')

        # Set the model to not have directory-based processing capability
        del self.inpainting_model.set_directories

        with patch.object(self.pipeline, 'process_image') as mock_process_image:
            self.pipeline.process_directory(self.input_dir, self.output_dir)

            expected_calls = [
                call(os.path.join(self.input_dir, 'image1.png'),
                     os.path.join(self.input_dir, 'image1_mask.png'),
                     os.path.join(self.output_dir, 'image1_inpainted.png')),
                call(os.path.join(self.input_dir, 'image2.jpg'),
                     os.path.join(self.input_dir, 'image2_mask.png'),
                     os.path.join(self.output_dir, 'image2_inpainted.jpg'))
            ]
            mock_process_image.assert_has_calls(expected_calls, any_order=True)

    @patch('framework.inpainting_pipeline.os.path.exists')
    @patch('framework.inpainting_pipeline.os.listdir')
    def test_process_directory_with_directory_model(self, mock_listdir, mock_exists):
        mock_listdir.return_value = ['image1.png', 'image2.jpg']
        mock_exists.side_effect = lambda x: x.endswith('_mask.png')
        self.inpainting_model.set_directories = MagicMock()
        self.inpainting_model.run_prediction = MagicMock()

        self.pipeline.process_directory(self.input_dir, self.output_dir)

        self.inpainting_model.set_directories.assert_called_once_with(indir=self.input_dir, outdir=self.output_dir)
        self.inpainting_model.run_prediction.assert_called_once()

    @patch('framework.inpainting_pipeline.os.path.exists')
    @patch('framework.inpainting_pipeline.os.listdir')
    def test_process_directory_skipping_images_without_masks(self, mock_listdir, mock_exists):
        mock_listdir.return_value = ['image1.png', 'image2.jpg', 'image3.jpeg']
        mock_exists.side_effect = lambda x: not x.endswith('image2_mask.png')

        # Set the model to not have directory-based processing capability
        del self.inpainting_model.set_directories

        with patch.object(self.pipeline, 'process_image') as mock_process_image:
            self.pipeline.process_directory(self.input_dir, self.output_dir)

            expected_calls = [
                call(os.path.join(self.input_dir, 'image1.png'),
                     os.path.join(self.input_dir, 'image1_mask.png'),
                     os.path.join(self.output_dir, 'image1_inpainted.png')),
                call(os.path.join(self.input_dir, 'image3.jpeg'),
                     os.path.join(self.input_dir, 'image3_mask.png'),
                     os.path.join(self.output_dir, 'image3_inpainted.jpeg'))
            ]
            mock_process_image.assert_has_calls(expected_calls, any_order=True)


if __name__ == '__main__':
    unittest.main()
