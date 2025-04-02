# [CVPR 2025] Exploiting Deblurring Networks for Radiance Fields (DeepDeblurRF)<br><sub>- Official PyTorch Implementation -</sub>

[![Project Website](https://img.shields.io/badge/Project-blue)](https://haeyun-choi.github.io/DDRF_page/)
[![arXiv](https://img.shields.io/badge/arXiv--b31b1b.svg)](https://arxiv.org/abs/2502.14454)
[![Dataset](https://img.shields.io/badge/Dataset-green)](https://drive.google.com/drive/folders/12t5J8VW102c2eXuj90RY9nVw5Uyv2YQ8)

**Exploiting Deblurring Networks for Radiance Fields**<br>
Haeyun Choi, Heemin Yang, Janghyuk Han, Sunghyun Cho<br>
KT, POSTECH

![Teaser image](./assets/teaser.png)

## Abstract
*In this paper, we propose DeepDeblurRF, a novel radiance field deblurring approach that can synthesize high-quality novel views from blurred training views with significantly reduced training time. DeepDeblurRF leverages deep neural network (DNN)-based deblurring modules to enjoy their deblurring performance and computational efficiency. To effectively combine DNN-based deblurring and radiance field construction, we propose a novel radiance field (RF)-guided deblurring and an iterative framework that performs RF-guided deblurring and radiance field construction in an alternating manner. Moreover, DeepDeblurRF is compatible with various scene representations, such as voxel grids and 3D Gaussians, expanding its applicability. We also present BlurRF-Synth, the first large-scale synthetic dataset for training radiance field deblurring frameworks. We conduct extensive experiments on both camera motion blur and defocus blur, demonstrating that DeepDeblurRF achieves state-of-the-art novel-view synthesis quality with significantly reduced training time.*

## News
* [2025-02-26] Our paper has been accepted to CVPR 2025! 🎉 
* [2025-02-27] Code & dataset will be released soon  
* [2025-04-02] Test dataset has been released! 🚀  
  - **BlurRF_Synth**, **BlurRF_Real**, **BlurRF_SB** are now available.  
  - Example Blender files for dataset generation are also provided.  
