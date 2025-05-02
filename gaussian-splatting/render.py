#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel


import imageio
import numpy as np

def render_video(model_path, name, iteration, video_cameras, gaussians, pipeline, background, fps=30):
    """Render a video using video cameras."""
    video_path = os.path.join(model_path, name, "ours_{}".format(iteration), "video.mp4")
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    
    frames = []
    for idx, camera in enumerate(video_cameras):
        # rendering = render(camera, gaussians, pipeline, background)["render"]
        rendering = torch.clamp(render(camera, gaussians, pipeline, background)["render"], 0.0, 1.0)
        frame = (rendering.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        frames.append(frame)
    
    # Save video with the specified FPS
    print(f"Saving video to {video_path} with FPS={fps}")
    imageio.mimwrite(video_path, frames, fps=fps, quality=8)
    print("Video saved successfully!")
    

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
 
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, video: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        bg_color = [1, 1, 1] if dataset.white_background else [0.5, 0.5, 0.5]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if video:
            render_video(dataset.model_path, "video", scene.loaded_iter, scene.getVideoCameras(), gaussians, pipeline, background, fps=args.fps)
            return

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)


   
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--video", action="store_true", help="Render a video using video_cameras")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the video")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.video)
