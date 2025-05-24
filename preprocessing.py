#!/usr/bin/env python
# coding: utf-8

# # DeepDeblurRF-G

# ## Initial Deblurring

# In[1]:


import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

print(torch.cuda.is_available())
print(torch.cuda.device_count())


# In[2]:


scene_type = "motion_dbnerf_real"  ################################ MODIFY THIS LINE -> "motion" or "motion_dbnerf_real", "defocus", "defocus_dbnerf_real" 
opt_path = f"./NAFNet/options/test/DDRF_G/{scene_type}/SD_NAFNet-width32.yml"

from basicsr.models import create_model
from basicsr.utils.options import parse
from basicsr.utils import img2tensor as _img2tensor, tensor2img, imwrite

opt = parse(opt_path, is_train=False)
opt['dist'] = False
NAFNet = create_model(opt)


# In[3]:


def imread(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def img2tensor(img, bgr2rgb=False, float32=True):
    img = img.astype(np.float32) / 255.
    return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)

def display(img1, img2):
    fig = plt.figure(figsize=(25, 10))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title('Input image', fontsize=16)
    ax1.axis('off')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title('NAFNet output', fontsize=16)
    ax2.axis('off')
    ax1.imshow(img1)
    ax2.imshow(img2)

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


# In[4]:


from PIL import Image

scene_name = "blurball" ################################ MODIFY THIS LINE 
index = 0

input_path = f'./data/{scene_name}/blur'
output_path = f'./data/{scene_name}/deblur/deblur_{index}'
os.makedirs(output_path, exist_ok=True)

for filename in os.listdir(input_path):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        core_name = filename[:-4]
        img_input = imread(os.path.join(input_path, filename))
        inp = img2tensor(img_input)
        img_output_path = os.path.join(output_path, core_name + '.png')
        single_image_inference(NAFNet, inp, img_output_path)


# In[5]:


import shutil

rf_path = f'./data/{scene_name}/rf/rf_{index}'
rf_images_path = os.path.join(rf_path, 'images')
os.makedirs(rf_images_path, exist_ok=True)

# Copy deblur_0 images
deblur_path = f'./data/{scene_name}/deblur/deblur_{index}'
for f in os.listdir(deblur_path):
    shutil.copy2(os.path.join(deblur_path, f), os.path.join(rf_images_path, f))

# Copy nv images
nv_path = f'./data/{scene_name}/nv'
for f in os.listdir(nv_path):
    shutil.copy2(os.path.join(nv_path, f), os.path.join(rf_images_path, f))

# Copy hold file
for f in os.listdir(f'./data/{scene_name}'):
    if f.startswith('hold'):
        shutil.copy2(os.path.join(f'./data/{scene_name}', f), os.path.join(rf_path, f))
        break


# In[6]:


# Restore basicsr_RF
nafnet_dir = "./NAFNet"
if os.path.exists(os.path.join(nafnet_dir, "basicsr")):
    os.rename(os.path.join(nafnet_dir, "basicsr"), os.path.join(nafnet_dir, "basicsr_SD"))
os.rename(os.path.join(nafnet_dir, "basicsr_RF"), os.path.join(nafnet_dir, "basicsr"))
print("Switched basicsr_RF → basicsr (for RF-guided deblurring)")


# ---
# 
# ## ✅ Proceed to `ddrf.py`
# 
# The initial preprocessing steps are now complete.
# 
# You can now run the main DDRF pipeline using:
# 
# ```bash
# python ddrf.py -c configs/<data_type>/<blur_type>or<None>/<scene_name>.txt  
# ```
# (e.g.,)  
# ```bash
# python ddrf.py -c configs/dbnerf_real/motion/blurball.txt

# 
