import os
import cv2
import sys
import time
import numpy as np
import shutil
import subprocess
import argparse
from PIL import Image
import torch
from imageio import imread
from basicsr.models import create_model
from basicsr.utils.options import parse
from basicsr.utils import img2tensor as _img2tensor, tensor2img, imwrite

def imread(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def img2tensor(img, bgr2rgb=False, float32=True):
    img = img.astype(np.float32) / 255.
    return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)

def single_image_inference(model, img, save_path):
    model.feed_data(data={'lq': img.unsqueeze(dim=0)})
    model.test()
    visuals = model.get_current_visuals()
    sr_img = tensor2img([visuals['result']])
    imwrite(sr_img, save_path)

def run_colmap_with_retries(imgs2poses_py, rf_folder, expected_images, retries=100):
    for attempt in range(retries):
        subprocess.run(['python', imgs2poses_py, rf_folder])
        if os.path.exists(os.path.join(rf_folder, 'poses_bounds.npy')):
            print(f"[COLMAP] Success on attempt {attempt+1} (poses_bounds.npy found).")
            return
        print(f"[COLMAP] Failed attempt {attempt+1}, retrying...")
        for item in ['sparse', 'colmap_output.txt', 'database.db']:
            path = os.path.join(rf_folder, item)
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif os.path.isfile(path):
                os.remove(path)
    raise RuntimeError(f"[COLMAP] Failed after {retries} attempts for {rf_folder}")

def extract_metrics(log_path):
    psnr, ssim, lpips = None, None, None
    if os.path.exists(log_path):
        with open(log_path) as f:
            lines = f.readlines()
        for line in lines:
            if "Evaluating test:" in line:
                parts = line.split()
                if "PSNR" in parts:
                    psnr = float(parts[parts.index("PSNR") + 1])
                if "SSIM" in parts:
                    ssim = float(parts[parts.index("SSIM") + 1])
                if "LPIPS" in parts:
                    lpips = float(parts[parts.index("LPIPS") + 1])
                if psnr and ssim and lpips:
                    break
    return psnr, ssim, lpips

def print_step(title):
    print("\n" + "=" * 60)
    print(f"[STEP] {title}")
    print("=" * 60 + "\n")
    sys.stdout.flush()

def prepare_rf_folder(scene_root, rf_index):
    rf_folder = os.path.join(scene_root, 'rf', f'rf_{rf_index}')
    deblur_input = os.path.join(scene_root, 'deblur', f'deblur_{rf_index}')
    os.makedirs(os.path.join(rf_folder, 'images'), exist_ok=True)

    for folder in ['deblur', 'nv']:
        src = deblur_input if folder == 'deblur' else os.path.join(scene_root, 'nv')
        for f in sorted(os.listdir(src)):
            shutil.copy(os.path.join(src, f), os.path.join(rf_folder, 'images', f))

    hold_file = [f for f in os.listdir(scene_root) if f.startswith('hold=')][0]
    shutil.copy(os.path.join(scene_root, hold_file), rf_folder)
    return rf_folder

def deblur_with_nafnet(scene_root, trviews_path, output_path, index, scene_type):
    opt_path = os.path.join('NAFNet', 'options', 'test', 'DDRF_G', scene_type, f'NAFNet-width64_{min(index, 4)}.yml')
    opt = parse(opt_path, is_train=False)
    opt['dist'] = False
    model = create_model(opt)

    input_path = os.path.join(scene_root, 'blur')
    os.makedirs(output_path, exist_ok=True)
    for input_img, rendered_img in zip(sorted(os.listdir(input_path)), sorted(os.listdir(trviews_path))):
        core_name = input_img[:-4]
        input_image = imread(os.path.join(input_path, input_img))
        rendered_image = imread(os.path.join(trviews_path, rendered_img))
        inp = torch.cat((img2tensor(input_image), img2tensor(rendered_image)), dim=0)
        single_image_inference(model, inp, os.path.join(output_path, core_name + '.png'))


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True)
args = parser.parse_args()

config = {}
with open(args.config) as f:
    exec(f.read(), config)

scene_name = config['scene_name']
scene_type = config['scene_type']
gpu_id = config.get('gpu', '0')
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

start_index = config.get('start_index', 1)
max_index = config['max_index']
iteration_list = config['iteration_list']

ddrf_root = './'
scene_root = os.path.join(ddrf_root, 'data', scene_name)
rendered_root = os.path.join(scene_root, 'rendered')
metrics_file = os.path.join(scene_root, 'metrics.txt')
imgs2poses_py = os.path.join(ddrf_root, 'LLFF', 'imgs2poses.py')
train_py = os.path.join(ddrf_root, 'gaussian-splatting', 'train.py')
render_py = os.path.join(ddrf_root, 'gaussian-splatting', 'render.py')

os.makedirs(rendered_root, exist_ok=True)
total_start_time = time.time()

for index in range(start_index, max_index + 1):
    print_step(f"Start Iteration {index}")
    start_time = time.time()
    iterations = iteration_list[index - 1]

    # === Step 1: Prepare input folder for COLMAP (Dᵢ₋₁ as input) ===
    rf_folder = prepare_rf_folder(scene_root, index - 1)

    # === Step 2: Run SfM via COLMAP using Dᵢ₋₁ ===
    image_dir = os.path.join(rf_folder, 'images')
    expected_images = len([f for f in os.listdir(image_dir) if f.endswith('.png')])
    run_colmap_with_retries(imgs2poses_py, rf_folder, expected_images)

    # === Step 3: Parse image IDs and split into train/test based on hold ===
    image_ids = sorted(int(os.path.splitext(f)[0]) for f in os.listdir(image_dir) if f.split('.')[0].isdigit())
    hold_val = int([f for f in os.listdir(scene_root) if f.startswith('hold=')][0].split('=')[-1])
    train_ids = [image_ids[i] for i in range(len(image_ids)) if i % hold_val != 0]
    test_ids = [image_ids[i] for i in range(len(image_ids)) if i % hold_val == 0]

    # === Step 4: Train 3D Gaussian Splatting model (RF from COLMAP structure) ===
    expname = f'{scene_name}_{index}'
    subprocess.run(['python', train_py, '--expname', expname, '-s', rf_folder,
                    '--port', '8888', '--eval', '--iterations', str(iterations),
                    '--test_iterations', str(iterations), '--save_iterations', str(iterations)])
    
    # === Step 5: Evaluate trained model and log quality metrics ===
    metrics_path = os.path.join("output", expname, "metrics_log.txt")
    psnr, ssim, lpips = extract_metrics(metrics_path)
    if psnr and ssim and lpips:
        with open(metrics_file, 'a') as f:
            f.write(f"[Iteration {index}] PSNR: {psnr:.2f} SSIM: {ssim:.4f} LPIPS: {lpips:.4f}\n")

    # === Step 6: Render novel views from the RF (Rᵢ) ===
    model_dir = os.path.join(ddrf_root, 'output', expname)
    subprocess.run(['python', render_py, '-m', model_dir, '--iteration', str(iterations), '--quiet'])

    # === Step 7: Organize rendered views into train/test sets ===
    trviews_path = os.path.join(rendered_root, f'trviews_{index}')
    tsviews_path = os.path.join(rendered_root, f'tsviews_{index}')
    os.makedirs(trviews_path, exist_ok=True)
    os.makedirs(tsviews_path, exist_ok=True)

    for mode, path, ids in [('train', trviews_path, train_ids), ('test', tsviews_path, test_ids)]:
        src = os.path.join(model_dir, mode, f'ours_{iterations}', 'renders')
        if os.path.isdir(src):
            files = sorted(os.listdir(src))
            assert len(files) == len(ids), f"[ERROR] Mismatch in {src}"
            for f, i in zip(files, ids):
                shutil.copy(os.path.join(src, f), os.path.join(path, f"{i:03d}.png"))

    # === Step 8: Save final output from last iteration ===
    if index == max_index:
        final_results_dir = os.path.join(scene_root, f'Final_results')
        os.makedirs(final_results_dir, exist_ok=True)
        for f in sorted(os.listdir(tsviews_path)):
            shutil.copy(os.path.join(tsviews_path, f), os.path.join(final_results_dir, f))

    # === Step 9: RF-guided Deblurring using [Blurry input + Rendered Rᵢ] ===
    if index < max_index:
        output_path = os.path.join(scene_root, 'deblur', f'deblur_{index}')
        print_step("Deblurring with NAFNet")
        deblur_with_nafnet(scene_root, trviews_path, output_path, index, scene_type)

    print(f"[Iteration {index}] Finished in {time.time() - start_time:.2f} seconds.")

print(f"[Total] Training finished in {(time.time() - total_start_time)/60:.2f} minutes.")
