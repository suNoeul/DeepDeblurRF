import os
import cv2
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

def run_colmap_with_retries(imgs2poses_py, rf_folder, expected_images, retries=10):
    for attempt in range(retries):
        subprocess.run(['python3', imgs2poses_py, rf_folder])
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
            if "Evaluating test:" in line and "PSNR" in line:
                parts = line.split()
                if "PSNR" in parts:
                    psnr = float(parts[parts.index("PSNR") + 1])
            if "Evaluating test:" in line and "SSIM" in line:
                parts = line.split()
                if "SSIM" in parts:
                    ssim = float(parts[parts.index("SSIM") + 1])
                if "LPIPS" in parts:
                    lpips = float(parts[parts.index("LPIPS") + 1])
            if psnr and ssim and lpips:
                break
    return psnr, ssim, lpips

# Load config
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True)
args = parser.parse_args()

config = {}
with open(args.config) as f:
    exec(f.read(), config)

max_index = config['max_index']
iteration_list = config['iteration_list']
scene_name = config['scene_name']
gpu_id = config.get('gpu', '0')
scene_type = config['scene_type']
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

ddrf_root = './'
scene_root = os.path.join(ddrf_root, 'data', scene_name)
rendered_root = os.path.join(scene_root, 'rendered')
metrics_file = os.path.join(scene_root, 'metrics.txt')
os.makedirs(rendered_root, exist_ok=True)

imgs2poses_py = os.path.join(ddrf_root, 'LLFF', 'imgs2poses.py')
train_py = os.path.join(ddrf_root, 'gaussian-splatting', 'train.py')
render_py = os.path.join(ddrf_root, 'gaussian-splatting', 'render.py')

hold_txt = [f for f in os.listdir(scene_root) if f.startswith('hold=')][0]
hold_val = int(hold_txt.split('=')[-1])

total_start_time = time.time()

for index in range(1, max_index + 1):
    start_time = time.time()
    iterations = iteration_list[index - 1]
    rf_input_index = index - 1
    rf_folder = os.path.join(scene_root, 'rf', f'rf_{rf_input_index}')
    deblur_input = os.path.join(scene_root, 'deblur', f'deblur_{rf_input_index}')

    print(f"[Iteration {index}] Starting pipeline...")

    os.makedirs(os.path.join(rf_folder, 'images'), exist_ok=True)
    for f in sorted(os.listdir(deblur_input)):
        shutil.copy(os.path.join(deblur_input, f), os.path.join(rf_folder, 'images', f))
    for f in sorted(os.listdir(os.path.join(scene_root, 'nv'))):
        shutil.copy(os.path.join(scene_root, 'nv', f), os.path.join(rf_folder, 'images', f))
    hold_file = [f for f in os.listdir(scene_root) if f.startswith('hold=')][0]
    shutil.copy(os.path.join(scene_root, hold_file), rf_folder)

    image_dir = os.path.join(rf_folder, 'images')
    expected_images = len([f for f in os.listdir(image_dir) if f.endswith('.png')])
    run_colmap_with_retries(imgs2poses_py, rf_folder, expected_images)

    images = sorted(os.listdir(image_dir))
    image_ids = [int(os.path.splitext(f)[0]) for f in images]
    train_ids = [i for i in image_ids if i % hold_val != 0]
    test_ids = [i for i in image_ids if i % hold_val == 0]

    expname = f'{scene_name}_{index}'
    command = [
        'python', train_py,
        '--expname', expname,
        '-s', rf_folder,
        '--port', '8888',
        '--eval', '--iterations', str(iterations),
        '--test_iterations', str(iterations),
        '--save_iterations', str(iterations)
    ]
    subprocess.run(command)

    log_path = os.path.join(scene_root, 'metrics_log.txt')
    psnr, ssim, lpips = extract_metrics(log_path)
    if psnr and ssim and lpips:
        with open(metrics_file, 'a') as f:
            f.write(f"[Iteration {index}] PSNR: {psnr:.2f}  SSIM: {ssim:.4f}  LPIPS: {lpips:.4f}\n")

    model_dir = os.path.join(ddrf_root, 'output', expname)
    render_iter = iterations
    render_cmd = ['python', render_py, '-m', model_dir, '--iteration', str(render_iter), '--quiet']
    subprocess.run(render_cmd)

    trviews_path = os.path.join(rendered_root, f'trviews_{index}')
    tsviews_path = os.path.join(rendered_root, f'tsviews_{index}')
    os.makedirs(trviews_path, exist_ok=True)
    os.makedirs(tsviews_path, exist_ok=True)
    train_render = os.path.join(model_dir, 'train', f'ours_{render_iter}', 'renders')
    test_render = os.path.join(model_dir, 'test', f'ours_{render_iter}', 'renders')

    for (src_dir, dst_dir, id_list) in [(train_render, trviews_path, train_ids), (test_render, tsviews_path, test_ids)]:
        if os.path.isdir(src_dir):
            for idx, f in enumerate(sorted(os.listdir(src_dir))):
                shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f"{id_list[idx]:03d}.png"))

    if index == max_index:
        final_results_dir = os.path.join(scene_root, f'Final_results')
        os.makedirs(final_results_dir, exist_ok=True)
        for f in sorted(os.listdir(tsviews_path)):
            shutil.copy(os.path.join(tsviews_path, f), os.path.join(final_results_dir, f))

    if index < max_index:
        opt_path = os.path.join(ddrf_root, 'NAFNet', 'options', 'test', 'DDRF_G', scene_type, f'NAFNet-width64_{min(index, 4)}.yml')
        opt = parse(opt_path, is_train=False)
        opt['dist'] = False
        NAFNet = create_model(opt)

        input_path = os.path.join(scene_root, 'blur')
        rendered_path = trviews_path
        output_path = os.path.join(scene_root, 'deblur', f'deblur_{index}')
        os.makedirs(output_path, exist_ok=True)

        input_images = sorted(os.listdir(input_path))
        rendered_images = sorted(os.listdir(rendered_path))

        for input_img, rendered_img in zip(input_images, rendered_images):
            core_name = input_img[:-4]
            input_image = os.path.join(input_path, input_img)
            rendered_image = os.path.join(rendered_path, rendered_img)
            img_input = imread(input_image)
            img_rendered = imread(rendered_image)
            inp_input = img2tensor(img_input)
            inp_rendered = img2tensor(img_rendered)
            inp = torch.cat((inp_input, inp_rendered), dim=0)
            img_output_path = os.path.join(output_path, core_name + '.png')
            single_image_inference(NAFNet, inp, img_output_path)

    elapsed = time.time() - start_time
    print(f"[Iteration {index}] Finished in {elapsed:.2f} seconds.")

total_elapsed = time.time() - total_start_time
print(f"[Total] Training finished in {total_elapsed / 60:.2f} minutes.")
