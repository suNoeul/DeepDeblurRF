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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

import numpy as np
from scipy.spatial.transform import Slerp, Rotation

from scene.cameras import Camera  # Camera 클래스 import

# def generate_video_cameras(train_cameras, num_frames=120):
#     """
#     Generate video cameras by interpolating between train_cameras.
    
#     Args:
#         train_cameras (list of Camera): Original train cameras.
#         num_frames (int): Total number of video frames.
        
#     Returns:
#         list of Camera: Interpolated video cameras.
#     """
#     video_cameras = []

#     # Extract R and T from train_cameras
#     R_list = [cam.R for cam in train_cameras]
#     T_list = [cam.T for cam in train_cameras]

#     # Convert R to Rotation objects for Slerp
#     rotations = Rotation.from_matrix(R_list)
#     slerp = Slerp(np.linspace(0, len(train_cameras) - 1, len(train_cameras)), rotations)

#     # Generate interpolation steps
#     interp_indices = np.linspace(0, len(train_cameras) - 1, num_frames)
#     interp_rotations = slerp(interp_indices)
#     interp_translations = np.array([
#         np.interp(interp_indices, np.arange(len(train_cameras)), T)
#         for T in np.array(T_list).T
#     ]).T

#     # Generate video_cameras
#     for idx, (R, T) in enumerate(zip(interp_rotations.as_matrix(), interp_translations)):
#         # Clone a base camera for properties like FOV, image, etc.
#         base_cam = train_cameras[0]
        
#         video_cameras.append(Camera(
#             colmap_id=base_cam.colmap_id,
#             R=R,
#             T=T,
#             FoVx=base_cam.FoVx,
#             FoVy=base_cam.FoVy,
#             image=base_cam.original_image.clone(),
#             gt_alpha_mask=None,
#             image_name=f"frame_{idx:04d}",
#             uid=idx,
#             data_device="cuda"
#         ))

#     return video_cameras

from scipy.interpolate import CubicSpline

def catmull_rom_spline(points, num_frames):
    """
    Generate a Catmull-Rom Spline path.
    
    Args:
        points (np.array): Control points (Nx3).
        num_frames (int): Number of frames to interpolate.
        
    Returns:
        np.array: Interpolated positions along the curve.
    """
    points = np.array(points)
    t = np.linspace(0, len(points) - 1, num_frames)
    x = CubicSpline(range(len(points)), points[:, 0], bc_type='natural')(t)
    y = CubicSpline(range(len(points)), points[:, 1], bc_type='natural')(t)
    z = CubicSpline(range(len(points)), points[:, 2], bc_type='natural')(t)
    return np.stack([x, y, z], axis=1)

from scipy.special import comb

def bezier_curve(points, num_frames):
    """
    Generate a Bezier curve path.
    
    Args:
        points (np.array): Control points (Nx3).
        num_frames (int): Number of frames to interpolate.
        
    Returns:
        np.array: Interpolated positions along the curve.
    """
    n = len(points) - 1  # Degree of the curve
    t = np.linspace(0, 1, num_frames)
    curve = np.zeros((num_frames, 3))
    for i in range(n + 1):
        binomial_coeff = comb(n, i)
        curve += binomial_coeff * (t**i)[:, None] * ((1 - t)**(n - i))[:, None] * points[i]
    return curve


# def generate_video_cameras(train_cameras, num_frames=120, method="linear"):
#     """
#     Generate video cameras by interpolating between train_cameras.
    
#     Args:
#         train_cameras (list of Camera): Original train cameras.
#         num_frames (int): Total number of video frames.
#         method (str): Interpolation method ("bezier", "catmull-rom", or "linear").
        
#     Returns:
#         list of Camera: Interpolated video cameras.
#     """
#     video_cameras = []

#     # Extract R and T from train_cameras
#     R_list = [cam.R for cam in train_cameras]
#     T_list = [cam.T for cam in train_cameras]

#     # Convert R to Rotation objects for Slerp
#     rotations = Rotation.from_matrix(R_list)
#     slerp = Slerp(np.linspace(0, len(train_cameras) - 1, len(train_cameras)), rotations)

#     # Generate interpolated translations
#     T_points = np.array(T_list)
#     if method == "bezier":
#         interp_translations = bezier_curve(T_points, num_frames)
#     elif method == "catmull-rom":
#         interp_translations = catmull_rom_spline(T_points, num_frames)
#     elif method == "linear":
#         interp_indices = np.linspace(0, len(train_cameras) - 1, num_frames)
#         interp_translations = np.array([
#             np.interp(interp_indices, np.arange(len(train_cameras)), T)
#             for T in T_points.T
#         ]).T
#     else:
#         raise ValueError("Invalid method. Choose 'bezier', 'catmull-rom', or 'linear'.")

#     # Generate interpolated rotations
#     interp_indices = np.linspace(0, len(train_cameras) - 1, num_frames)
#     interp_rotations = slerp(interp_indices)

#     # Create video cameras
#     for idx, (R, T) in enumerate(zip(interp_rotations.as_matrix(), interp_translations)):
#         base_cam = train_cameras[0]
#         video_cameras.append(Camera(
#             colmap_id=base_cam.colmap_id,
#             R=R,
#             T=T,
#             FoVx=base_cam.FoVx,
#             FoVy=base_cam.FoVy,
#             image=base_cam.original_image.clone(),
#             gt_alpha_mask=None,
#             image_name=f"frame_{idx:04d}",
#             uid=idx,
#             data_device="cuda"
#         ))

#     return video_cameras



def generate_video_cameras(train_cameras, num_frames=120, method="linear"):
    """
    Generate video cameras by interpolating between selected train_cameras.
    
    Args:
        train_cameras (list of Camera): Original train cameras.
        num_frames (int): Total number of video frames.
        method (str): Interpolation method ("bezier", "catmull-rom", or "linear").
        
    Returns:
        list of Camera: Interpolated video cameras.
    """
    # video_cameras = []
    # num_segments = 5  # Divide into 5 segments, resulting in 6 key cameras
    # segment_frames = num_frames // num_segments

    # # Select key cameras (0, n, 2n, ..., 5n)
    # key_cameras = [train_cameras[i * (len(train_cameras) // num_segments)] for i in range(num_segments + 1)]

    
    
    ##############################################
    num_segments = 5 
    if len(train_cameras) < num_segments + 1:
        raise ValueError(f"train_cameras should have at least {num_segments + 1} elements, but got {len(train_cameras)}.")

    video_cameras = []
    segment_frames = num_frames // num_segments

    # Select key cameras (first, intermediate points, and last)
    step = len(train_cameras) / num_segments
    key_cameras = [train_cameras[round(i * step)] for i in range(num_segments)]
    key_cameras.append(train_cameras[-1])  # 마지막 카메라는 리스트의 마지막 요소로 설정
    
    

    # Extract R and T from selected key cameras
    R_list = [cam.R for cam in key_cameras]
    T_list = [cam.T for cam in key_cameras]

    # Convert R to Rotation objects for Slerp
    rotations = Rotation.from_matrix(R_list)
    slerp = Slerp(np.linspace(0, len(key_cameras) - 1, len(key_cameras)), rotations)

    # Generate interpolated translations
    T_points = np.array(T_list)
    if method == "bezier":
        interp_translations = bezier_curve(T_points, num_frames)
    elif method == "catmull-rom":
        interp_translations = catmull_rom_spline(T_points, num_frames)
    elif method == "linear":
        interp_indices = np.linspace(0, len(key_cameras) - 1, num_frames)
        interp_translations = np.array([
            np.interp(interp_indices, np.arange(len(key_cameras)), T)
            for T in T_points.T
        ]).T
    else:
        raise ValueError("Invalid method. Choose 'bezier', 'catmull-rom', or 'linear'.")

    # Generate interpolated rotations
    interp_indices = np.linspace(0, len(key_cameras) - 1, num_frames)
    interp_rotations = slerp(interp_indices)

    # Create video cameras
    for idx, (R, T) in enumerate(zip(interp_rotations.as_matrix(), interp_translations)):
        base_cam = train_cameras[0]
        video_cameras.append(Camera(
            colmap_id=base_cam.colmap_id,
            R=R,
            T=T,
            FoVx=base_cam.FoVx,
            FoVy=base_cam.FoVy,
            image=base_cam.original_image.clone(),
            gt_alpha_mask=None,
            image_name=f"frame_{idx:04d}",
            uid=idx,
            data_device="cuda"
        ))

    return video_cameras


class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        
        self.video_cameras = {}
        
        print(args.source_path)

        if os.path.exists(os.path.join(args.source_path, "sparse")):
############################################### Follow the Deblurring-3D-Gaussian-Splatting/scene/__init__.py ###############################################
            li = os.listdir(args.source_path)
            llffhold = 8
            for l in li:
                if l.startswith("hold"):
                    llffhold = int(l.split("=")[-1])
                    break
            print("TEST VIEW HOLD: ", llffhold)
#############################################################################################################################################################
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, llffhold)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"
            
        # import pdb
        # pdb.set_trace()
        
        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            
            
            print("Loading Video Cameras")
            self.video_cameras[resolution_scale] = generate_video_cameras(self.train_cameras[resolution_scale], num_frames=120, method="bezier")
            

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)   
        

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def getVideoCameras(self, scale=1.0):
        return self.video_cameras[scale]