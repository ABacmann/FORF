import json


class Config:
    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            config_data = json.load(file)

        self.sam_ckpt_path = config_data['sam_ckpt_path']
        self.sam_segment_script = config_data['sam_segment_script']
        self.sam_model_type = config_data['sam_model_type']
        self.dilate_kernel_size = config_data['dilate_kernel_size']
        self.point_labels = config_data['point_labels']

