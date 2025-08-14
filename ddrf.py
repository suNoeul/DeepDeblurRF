import os
import cv2
import time
import numpy as np
import shutil
import subprocess
import argparse
import torch

from basicsr.models import create_model
from basicsr.utils.options import parse
from basicsr.utils import img2tensor as _img2tensor, tensor2img, imwrite

# -----------------------------
# Utility Functions
# -----------------------------
def read_image(path):
    """Read image (RGB)."""
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def img2tensor(img, bgr2rgb=False, float32=True):
    """Convert numpy image to tensor (0~1)."""
    img = img.astype(np.float32) / 255.
    return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)

def run_inference(model, img, save_path):
    """Run single-image inference and save result."""
    model.feed_data({'lq': img.unsqueeze(0)})
    model.test()
    result = tensor2img([model.get_current_visuals()['result']])
    imwrite(result, save_path)

def run_colmap(imgs2poses_py, rf_folder, expected_images, retries=100):
    """Run COLMAP with retries until poses are generated."""
    for attempt in range(retries):
        subprocess.run(['python', imgs2poses_py, rf_folder])
        if os.path.exists(os.path.join(rf_folder, 'poses_bounds.npy')):
            print(f"[COLMAP] Success on attempt {attempt+1}.")
            return
        print(f"[COLMAP] Failed attempt {attempt+1}, retrying...")
        for item in ['sparse', 'colmap_output.txt', 'database.db']:
            p = os.path.join(rf_folder, item)
            if os.path.isdir(p):
                shutil.rmtree(p)
            elif os.path.isfile(p):
                os.remove(p)
    raise RuntimeError(f"[COLMAP] Failed after {retries} attempts.")

def parse_metrics(log_path):
    """Extract PSNR/SSIM/LPIPS from log file."""
    psnr = ssim = lpips = None
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                if "Evaluating test:" in line:
                    parts = line.split()
                    if "PSNR" in parts: psnr = float(parts[parts.index("PSNR") + 1])
                    if "SSIM" in parts: ssim = float(parts[parts.index("SSIM") + 1])
                    if "LPIPS" in parts: lpips = float(parts[parts.index("LPIPS") + 1])
                if psnr and ssim and lpips:
                    break
    return psnr, ssim, lpips

# -----------------------------
# Main Pipeline
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    args = parser.parse_args()

    # Load config (simple exec-based load)
    cfg = {}
    with open(args.config) as f:
        exec(f.read(), cfg)

    start_idx = cfg.get('start_index', 1)
    max_idx = cfg['max_index']
    iteration_list = cfg['iteration_list']
    scene_name = cfg['scene_name']
    gpu_id = cfg.get('gpu', '0')
    scene_type = cfg['scene_type']

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    ddrf_root = './'
    scene_root = os.path.join(ddrf_root, 'data', scene_name)
    rendered_root = os.path.join(scene_root, 'rendered')
    metrics_file = os.path.join(scene_root, 'metrics.txt')
    os.makedirs(rendered_root, exist_ok=True)

    imgs2poses_py = os.path.join(ddrf_root, 'LLFF', 'imgs2poses.py')
    train_py = os.path.join(ddrf_root, 'gaussian-splatting', 'train.py')
    render_py = os.path.join(ddrf_root, 'gaussian-splatting', 'render.py')

    hold_val = int([f for f in os.listdir(scene_root) if f.startswith('hold=')][0].split('=')[-1])
    total_start = time.time()

    for idx in range(start_idx, max_idx + 1):
        print(f"\n[Iteration {idx}] Starting pipeline...")
        iter_start = time.time()

        rf_input_idx = idx - 1
        rf_folder = os.path.join(scene_root, 'rf', f'rf_{rf_input_idx}')
        deblur_input = os.path.join(scene_root, 'deblur', f'deblur_{rf_input_idx}')

        # Prepare COLMAP input images
        os.makedirs(os.path.join(rf_folder, 'images'), exist_ok=True)
        for folder in [deblur_input, os.path.join(scene_root, 'nv')]:
            for f in sorted(os.listdir(folder)):
                shutil.copy(os.path.join(folder, f), os.path.join(rf_folder, 'images', f))
        shutil.copy([os.path.join(scene_root, f) for f in os.listdir(scene_root) if f.startswith('hold=')][0], rf_folder)

        # Run COLMAP
        expected_imgs = len([f for f in os.listdir(os.path.join(rf_folder, 'images')) if f.endswith('.png')])
        run_colmap(imgs2poses_py, rf_folder, expected_imgs)

        # Train RF model
        expname = f'{scene_name}_{idx}'
        iterations = iteration_list[idx - 1]
        subprocess.run([
            'python', train_py,
            '--expname', expname, '-s', rf_folder,
            '--port', '8888', '--eval',
            '--iterations', str(iterations),
            '--test_iterations', str(iterations),
            '--save_iterations', str(iterations)
        ])

        # Log metrics
        psnr, ssim, lpips = parse_metrics(os.path.join(scene_root, 'metrics_log.txt'))
        if psnr and ssim and lpips:
            with open(metrics_file, 'a') as f:
                f.write(f"[Iteration {idx}] PSNR: {psnr:.2f}  SSIM: {ssim:.4f}  LPIPS: {lpips:.4f}\n")

        # Render RF outputs
        model_dir = os.path.join(ddrf_root, 'output', expname)
        subprocess.run(['python', render_py, '-m', model_dir, '--iteration', str(iterations), '--quiet'])

        # Organize rendered images
        trviews_path = os.path.join(rendered_root, f'trviews_{idx}')
        tsviews_path = os.path.join(rendered_root, f'tsviews_{idx}')
        os.makedirs(trviews_path, exist_ok=True)
        os.makedirs(tsviews_path, exist_ok=True)

        train_render = os.path.join(model_dir, 'train', f'ours_{iterations}', 'renders')
        test_render = os.path.join(model_dir, 'test', f'ours_{iterations}', 'renders')

        image_ids = sorted(int(os.path.splitext(f)[0]) for f in os.listdir(os.path.join(rf_folder, 'images')) if f.split('.')[0].isdigit())
        train_ids = [i for i in image_ids if i % hold_val != 0]
        test_ids = [i for i in image_ids if i % hold_val == 0]

        for (src, dst, ids) in [(train_render, trviews_path, train_ids), (test_render, tsviews_path, test_ids)]:
            if os.path.isdir(src):
                files = sorted(os.listdir(src))
                assert len(files) == len(ids), f"[ERROR] Mismatch: {len(files)} files vs {len(ids)} IDs"
                for f_render, true_id in zip(files, ids):
                    shutil.copy(os.path.join(src, f_render), os.path.join(dst, f"{true_id:03d}.png"))

        # Save final results
        if idx == max_idx:
            final_dir = os.path.join(scene_root, 'Final_results')
            os.makedirs(final_dir, exist_ok=True)
            for f in sorted(os.listdir(tsviews_path)):
                shutil.copy(os.path.join(tsviews_path, f), os.path.join(final_dir, f))

        # RF-guided deblurring for next iteration
        if idx < max_idx:
            opt_path = os.path.join(ddrf_root, 'NAFNet', 'options', 'test', 'DDRF_G', scene_type, f'NAFNet-width64_{min(idx, 4)}.yml')
            opt = parse(opt_path, is_train=False)
            opt['dist'] = False
            NAFNet = create_model(opt)

            input_path = os.path.join(scene_root, 'blur')
            rendered_path = trviews_path
            output_path = os.path.join(scene_root, 'deblur', f'deblur_{idx}')
            os.makedirs(output_path, exist_ok=True)

            for in_img, rend_img in zip(sorted(os.listdir(input_path)), sorted(os.listdir(rendered_path))):
                core = in_img[:-4]
                inp_input = img2tensor(read_image(os.path.join(input_path, in_img)))
                inp_render = img2tensor(read_image(os.path.join(rendered_path, rend_img)))
                combined = torch.cat((inp_input, inp_render), dim=0)
                run_inference(NAFNet, combined, os.path.join(output_path, core + '.png'))

        print(f"[Iteration {idx}] Done in {time.time() - iter_start:.2f}s")

    print(f"\n[Total] Finished in {(time.time() - total_start)/60:.2f} min.")

if __name__ == "__main__":
    main()
