#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import torch
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from basicsr.models import create_model
from basicsr.utils.options import parse
from basicsr.utils import img2tensor as _img2tensor, tensor2img, imwrite

# MODIFY HERE: List of (scene_name, scene_type) tuples
scenes = [
    ("blurball", "motion_dbnerf_real"),
    ("deskchair", "motion_dbnerf_real"),
    ("nestchair", "motion_dbnerf_real"),
    ("cozyroom", "defocus"),
    ("pool", "defocus"),
    ("shotgun", "defocus")
]

index = 0  # iteration index

# Image utilities
def imread(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def img2tensor(img, bgr2rgb=False, float32=True):
    img = img.astype(np.float32) / 255.
    return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)

def single_image_inference(model, img, save_path):
    model.feed_data(data={'lq': img.unsqueeze(dim=0)})
    if model.opt['val'].get('grids', False):
        model.grids()
    model.test()
    if model.opt['val'].get('grids', False):
        model.grids_inverse()
    visuals = model.get_current_visuals()
    sr_img = tensor2img([visuals['result']])
    imwrite(sr_img, save_path)

# Process each scene
for scene_name, scene_type in scenes:
    print(f"\n=== Preprocessing: {scene_name} ({scene_type}) ===")

    # Load NAFNet model for this scene_type
    opt_path = f"./NAFNet/options/test/DDRF_G/{scene_type}/SD_NAFNet-width32.yml"
    opt = parse(opt_path, is_train=False)
    opt['dist'] = False
    NAFNet = create_model(opt)

    # Step 1: Deblur
    input_path = f'./data/{scene_name}/blur'
    output_path = f'./data/{scene_name}/deblur/deblur_{index}'
    os.makedirs(output_path, exist_ok=True)

    print(f"[{scene_name}] Deblurring input images in: {input_path}")
    for filename in sorted(os.listdir(input_path)):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            core_name = os.path.splitext(filename)[0]
            img_input = imread(os.path.join(input_path, filename))
            inp = img2tensor(img_input)
            img_output_path = os.path.join(output_path, f"{core_name}.png")
            single_image_inference(NAFNet, inp, img_output_path)

    # Step 2: Prepare RF input
    rf_path = f'./data/{scene_name}/rf/rf_{index}'
    rf_images_path = os.path.join(rf_path, 'images')
    os.makedirs(rf_images_path, exist_ok=True)

    # Copy deblurred images
    for f in os.listdir(output_path):
        shutil.copy2(os.path.join(output_path, f), os.path.join(rf_images_path, f))

    # Copy nv/ images
    nv_path = f'./data/{scene_name}/nv'
    for f in os.listdir(nv_path):
        shutil.copy2(os.path.join(nv_path, f), os.path.join(rf_images_path, f))

    # Copy hold file
    for f in os.listdir(f'./data/{scene_name}'):
        if f.startswith('hold'):
            shutil.copy2(os.path.join(f'./data/{scene_name}', f), os.path.join(rf_path, f))
            break

    print(f"[{scene_name}] RF input images prepared in: {rf_images_path}")

# Step 3: Switch basicsr directory from SD to RF (only once)
nafnet_dir = "./NAFNet"
sd_dir = os.path.join(nafnet_dir, "basicsr")
rf_dir = os.path.join(nafnet_dir, "basicsr_RF")
sd_backup = os.path.join(nafnet_dir, "basicsr_SD")

if os.path.exists(sd_dir):
    os.rename(sd_dir, sd_backup)
os.rename(rf_dir, sd_dir)

print("\nSwitched basicsr_RF → basicsr (for RF-guided deblurring)")
print("All preprocessing complete.")
