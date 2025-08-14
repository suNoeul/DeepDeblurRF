import os
import torch
import cv2
import numpy as np
import shutil

# --- NAFNet/basicsr imports ---
# Ensure your environment is set up for these imports to work
from basicsr.models import create_model
from basicsr.utils.options import parse
from basicsr.utils import img2tensor as _img2tensor, tensor2img, imwrite

# ==============================================================================
# 1. USER CONFIGURATION
# ==============================================================================

# MODIFY THIS -> "motion", "motion_dbnerf_real", "defocus", "defocus_dbnerf_real"
scene_type = "motion_dbnerf_real"

# MODIFY THIS -> e.g., "blurball", "stair", etc.
scene_name = "blurball"

# Index for the deblurring/RF run
index = 0

# ==============================================================================
# Helper Functions
# ==============================================================================

def imread(img_path):
    """Reads an image and converts it from BGR to RGB."""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def img2tensor(img, bgr2rgb=False, float32=True):
    """Converts a numpy image to a PyTorch tensor."""
    img = img.astype(np.float32) / 255.
    return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)

def single_image_inference(model, img_tensor, save_path):
    """Runs the NAFNet model on a single image tensor and saves the output."""
    model.feed_data(data={'lq': img_tensor.unsqueeze(dim=0)})

    if model.opt['val'].get('grids', False):
        model.grids()

    model.test()

    if model.opt['val'].get('grids', False):
        model.grids_inverse()

    visuals = model.get_current_visuals()
    sr_img = tensor2img([visuals['result']])
    imwrite(sr_img, save_path)

# ==============================================================================
# Main Execution Logic
# ==============================================================================

if __name__ == '__main__':
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    print("-" * 50)

    # --- Part 1: Initial Deblurring ---
    print("Step 1: Performing initial deblurring with NAFNet...")

    # Initialize NAFNet model
    opt_path = f"./NAFNet/options/test/DDRF_G/{scene_type}/SD_NAFNet-width32.yml"
    opt = parse(opt_path, is_train=False)
    opt['dist'] = False
    NAFNet = create_model(opt)
    
    # Define paths
    input_path = f'./data/{scene_name}/blur'
    output_path = f'./data/{scene_name}/deblur/deblur_{index}'
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Reading images from: {input_path}")
    print(f"Saving deblurred images to: {output_path}")

    # Process each image in the input directory
    image_files = [f for f in os.listdir(input_path) if f.endswith(('.png', '.jpg'))]
    for filename in image_files:
        core_name = os.path.splitext(filename)[0]
        print(f"  Processing {filename}...")
        
        img_input_path = os.path.join(input_path, filename)
        img_input = imread(img_input_path)
        inp_tensor = img2tensor(img_input)
        
        img_output_path = os.path.join(output_path, core_name + '.png')
        single_image_inference(NAFNet, inp_tensor, img_output_path)

    print("Initial deblurring complete.")
    print("-" * 50)

    # --- Part 2: Prepare RF Directory Structure ---
    print("Step 2: Preparing directory structure for RF-guided deblurring...")
    
    rf_path = f'./data/{scene_name}/rf/rf_{index}'
    rf_images_path = os.path.join(rf_path, 'images')
    os.makedirs(rf_images_path, exist_ok=True)

    # Copy deblurred images
    deblur_path = f'./data/{scene_name}/deblur/deblur_{index}'
    for f in os.listdir(deblur_path):
        shutil.copy2(os.path.join(deblur_path, f), os.path.join(rf_images_path, f))
    print(f"  Copied deblurred images to {rf_images_path}")

    # Copy 'nv' images (if they exist)
    nv_path = f'./data/{scene_name}/nv'
    if os.path.exists(nv_path):
        for f in os.listdir(nv_path):
            shutil.copy2(os.path.join(nv_path, f), os.path.join(rf_images_path, f))
        print(f"  Copied 'nv' images to {rf_images_path}")

    # Copy 'hold' file
    for f in os.listdir(f'./data/{scene_name}'):
        if f.startswith('hold'):
            shutil.copy2(os.path.join(f'./data/{scene_name}', f), os.path.join(rf_path, f))
            print(f"  Copied '{f}' file to {rf_path}")
            break
            
    print("Directory preparation complete.")
    print("-" * 50)
    
    # --- Part 3: Switch basicsr Library ---
    print("Step 3: Switching basicsr library version for next step...")
    nafnet_dir = "./NAFNet"
    basicsr_active_path = os.path.join(nafnet_dir, "basicsr")
    basicsr_sd_path = os.path.join(nafnet_dir, "basicsr_SD")
    basicsr_rf_path = os.path.join(nafnet_dir, "basicsr_RF")

    if os.path.exists(basicsr_active_path):
        os.rename(basicsr_active_path, basicsr_sd_path)
    
    if os.path.exists(basicsr_rf_path):
        os.rename(basicsr_rf_path, basicsr_active_path)
        print("Switched basicsr_RF -> basicsr (for RF-guided deblurring)")
    else:
        print(f"Warning: Could not find '{basicsr_rf_path}' to switch to.")

    print("-" * 50)
    print("\n✅ Preprocessing complete.")
    print("You can now run the main DDRF pipeline, for example:")
    print(f"   python ddrf.py -c configs/dbnerf_real/motion/{scene_name}.txt\n")