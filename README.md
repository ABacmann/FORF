# Framework for Anonymizing Medical Images

## Overview

Ensuring patient privacy while maintaining the integrity of medical records is paramount in the evolving landscape of medical data management. This framework addresses this challenge by providing a robust system for anonymizing medical images, specifically targeting the removal of identifiable foreign objects such as implants or pacemakers. The framework leverages the LaMa inpainting model by Surovov et al. ([repo](https://github.com/advimman/lama)), including fine-tuned versions on chest X-ray images, and the deepfillv2_pytorch model ([repo](https://github.com/nipponjo/deepfillv2-pytorch)). Utilizing these advanced image inpainting techniques with Convolutional Neural Networks (CNNs) and Generative Adversarial Networks (GANs), the framework ensures patient privacy without compromising the utility of medical datasets. It also incorporates the Segment Anything Model (SAM) for mask generation, sourced from the Inpaint-Anything project ([repo](https://github.com/geekyutao/Inpaint-Anything)). The framework complies with stringent data protection regulations, particularly those in Switzerland, ensuring sensitive patient data is handled correctly and securely.

## Installation Guidelines

This section provides detailed instructions for installing and setting up the framework used in this project. The framework relies on pre-trained model files that need to be downloaded and placed in specific directories. Follow the steps below to ensure proper installation and configuration.

## Step 1: Clone the GitHub Repository

First, clone the GitHub repository containing the framework code to your local machine. Open your terminal and execute the following commands:

```bash
git clone https://github.com/ABacmann/FORF.git
cd FORF
```

## Step 2: Create the Conda Environment

The framework requires several Python packages. These can be installed using a conda environment. Execute the following commands to create the environment from the provided `environment.yml` file located in the framework directory:

```bash
conda env create -f framework/environment.yml
conda activate <your-environment-name>
```
Replace <your-environment-name> with the name of the environment specified in the environment.yml file.

## Step 3: Download Pre-trained Weights

The framework uses pre-trained weights for its deep learning models. These weights need to be downloaded from ([this link](https://uzh-my.sharepoint.com/personal/alexandrequentinpol_bacmann_uzh_ch/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Falexandrequentinpol%5Fbacmann%5Fuzh%5Fch%2FDocuments%2FModels&ga=1)).

### DeepFillv2 Model

1. Navigate to the deepfillv2 folder in the provided link.
2. Download the file `states_pt_places2.pth`.
3. Move the downloaded file to the `pre-trained` folder within the `deepfillv2_pytorch` directory.

```bash
mv path/to/downloaded/states_pt_places2.pth path/to/your/repo/deepfillv2_pytorch/pretrained
```

### LaMa Model

1. Navigate to the `LaMa` folder in the provided link.
2. Download all three subfolders within the `LaMa` folder.
3. Move the downloaded folders to the `lama/models` directory:

```bash
mv path/to/downloaded/big-lama path/to/your/repo/lama/models
mv path/to/downloaded/fine-tuned-big-lama path/to/your/repo/lama/models
mv path/to/downloaded/fine-tuned2-big-lama path/to/your/repo/lama/models
```

### SAM Model

1. Navigate to the `SAM` folder in the provided link.
2. Download the files `sam_vit_h_4b8939.pth` and `sttn.pth`.
3. Move these files to the `Inpaint-Anything/pretrained_models` directory:

```bash
mv path/to/downloaded/sam_vit_h_4b8939.pth path/to/your/repo/Inpaint-Anything/pretrained_models
mv path/to/downloaded/sttn.pth path/to/your/repo/Inpaint-Anything/pretrained_models
```

## Step 4: Set Model Paths

For the inpainting model, paths need to be set in the configuration files located in the `config` folder. Update the paths in the following configuration files:

### config/.env

```bash
TORCH_HOME=path/to/your/repo/lama
BIG_LAMA_MODEL_CONFIG=path/to/your/repo/config/big_lama_model_config.yaml
FINE_TUNED_LAMA_MODEL_CONFIG=path/to/your/repo/config/fine_tuned_lama_model_config.yaml
FINE_TUNED_LAMA_MODEL2_CONFIG=path/to/your/repo/config/fine_tuned_lama_model2_config.yaml
```

### config/big_lama_model_config.yaml

```bash
model_dir: path/to/your/repo/lama/models/big-lama
config_path: path/to/your/repo/lama/models/big-lama/config.yaml
checkpoint: best.ckpt
```

### config/fine_tuned_lama_model2_config.yaml

```bash
model_dir: path/to/your/repo/lama/models/fine-tuned-big-lama
config_path: path/to/your/repo/lama/models/fine-tuned-big-lama/config.yaml
checkpoint: fine-tuned-epoch12.ckpt
```
### framework/mask_generator/config.json

```bash
{
    "sam_ckpt_path": "path/to/your/repo/Inpaint-Anything/pretrained_models/sam_vit_h_4b8939.pth",
    "sam_segment_script": "path/to/your/repo/Inpaint-Anything/sam_segment.py",
    "sam_model_type": "vit_h",
    "dilate_kernel_size": 15,
    "point_labels": 1
}
```

### framework/models/Deep_Fill_V2.py
Set the deepfillv2 model path in the deepfillv2 class

```bash
self.model_path = path/to/your/repo/deepfillv2_pytorch/pretrained/states_pt_places2.pth
```

## Step 5: Run the Framework

To run the framework, you can use the `run_pipeline()` function in the main module. 

By following these steps, you should be able to set up and run the deep learning framework successfully.

## How to Use the Framework

1. **Select a File or Folder:**
   - Begin by selecting the file or folder containing the image(s) from which you want to remove an object.

2. **Choose a Save Location:**
   - Select a folder where the processed images will be saved.

3. **Select Mask Generation Method:**
   - You have two options for mask generation: Manual Mask Generation or SAM Mask Generation.

4. **Manual Mask Generation:**
   - If you choose manual mask generation:
     - A graphical user interface (GUI) will be provided.
     - Draw a mask over the object you wish to remove directly on the image.

5. **SAM Mask Generation:**
   - If you opt for SAM (Segment Anything Model) mask generation:
     - Click on the object in the image to create the mask.
     - Right-click to finalize the selection and proceed.
     - Note that it can take a while if used with a CPU.

6. **Choose Anonymization Method:**
   - After generating the mask, select the desired anonymization method to apply to the masked area.

7. **Process and Save:**
   - Once all selections are made, the framework will process the image(s) according to the chosen options.
   - The processed images will be saved in the folder specified in step 2.

By following these steps, you can effectively remove objects from images and apply anonymization as needed using the framework. This ensures patient privacy while maintaining the utility of medical datasets.





