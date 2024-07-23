# This is deepfillv2-pytorch/model/__init__.py

from .networks import Generator  # Import the Generator class from networks.py
from .networks_tf import Generator as GeneratorTF  # Import the Generator class from networks_tf.py

import torch


def load_model(path, device='cuda'):
    try:
        # Use map_location to load the model on the specified device (CPU or GPU)
        gen_sd = torch.load(path, map_location=torch.device(device))['G']
    except FileNotFoundError:
        return None

    # Determine which model to instantiate based on the keys in the state dictionary
    if 'stage1.conv1.conv.weight' in gen_sd.keys():
        model = Generator()
    else:
        model = GeneratorTF()

    # Move the model to the specified device
    model = model.to(device)
    model.eval()

    # Load the state dictionary into the model
    model.load_state_dict(gen_sd, strict=False)

    return model
