# [CVPR 2025] Exploiting Deblurring Networks for Radiance Fields (DeepDeblurRF)<br><sub>- Official PyTorch Implementation -</sub>  
*Fast and high-quality novel view synthesis from blurry images using iterative deblurring and radiance field construction.*

[![Project Website](https://img.shields.io/badge/Project--blue)](https://haeyun-choi.github.io/DDRF_page/)
[![arXiv](https://img.shields.io/badge/arXiv--b31b1b.svg)](https://arxiv.org/abs/2502.14454)
[![Dataset](https://img.shields.io/badge/Dataset--green)](https://drive.google.com/drive/folders/12t5J8VW102c2eXuj90RY9nVw5Uyv2YQ8)

**Exploiting Deblurring Networks for Radiance Fields**  
Haeyun Choi, Heemin Yang, Janghyuk Han, Sunghyun Cho  
KT, POSTECH

## Overview
![Teaser image](./assets/teaser.png)

## Abstract
*In this paper, we propose DeepDeblurRF, a novel radiance field deblurring approach that can synthesize high-quality novel views from blurred training views with significantly reduced training time. DeepDeblurRF leverages deep neural network (DNN)-based deblurring modules to enjoy their deblurring performance and computational efficiency. To effectively combine DNN-based deblurring and radiance field construction, we propose a novel radiance field (RF)-guided deblurring and an iterative framework that performs RF-guided deblurring and radiance field construction in an alternating manner. Moreover, DeepDeblurRF is compatible with various scene representations, such as voxel grids and 3D Gaussians, expanding its applicability. We also present BlurRF-Synth, the first large-scale synthetic dataset for training radiance field deblurring frameworks. We conduct extensive experiments on both camera motion blur and defocus blur, demonstrating that DeepDeblurRF achieves state-of-the-art novel-view synthesis quality with significantly reduced training time.*

## News  
[Feb 26, 2025] Our paper has been accepted to CVPR 2025! 🎉  
[Feb 27, 2025] Code & dataset will be released soon  
[Apr 2, 2025] Test dataset has been released! 🚀

## Iterative Pipeline Structure in DeepDeblurRF

DeepDeblurRF is built as an iterative pipeline that progressively refines novel view synthesis from blurry inputs. Each iteration consists of radiance field (RF) construction and RF-guided deblurring. 

The structure below summarizes the process:

| Iteration (`index`) | RF Input Folder | Rendered Views (train/test) | NAFNet Settings           | RF-Guided Deblur Output | Final Output         |
|---------------------|------------------|-------------------------------|----------------------------|--------------------------|-----------------------|
| 1                   | `rf_0`           | `trviews_1`, `tsviews_1`     | `NAFNet-width64_1.yml`     | `deblur_1`               | ❌                    |
| 2                   | `rf_1`           | `trviews_2`, `tsviews_2`     | `NAFNet-width64_2.yml`     | `deblur_2`               | ❌                    |
| 3                   | `rf_2`           | `trviews_3`, `tsviews_3`     | `NAFNet-width64_3.yml`     | `deblur_3`               | ❌                    |
| 4                   | `rf_3`           | `trviews_4`, `tsviews_4`     | `NAFNet-width64_4.yml`     | `deblur_4`               | ❌                    |
| 5                   | `rf_4`           | `trviews_5`, `tsviews_5`     | *(skipped)*                | *(not created)*          | ✅ `tsviews_5`         |

- Iteration starts from `index = 1`. Each RF input `rf_{index-1}` is constructed from `deblur_{index-1}` and novel views (`nv`).
- The initial deblur input `deblur_0` is generated beforehand via `preprocess.ipynb`.
- Rendered views `trviews_{index}` and `tsviews_{index}` are generated from the trained RF.
- RF-guided deblurring is applied using the original `blur` images and the rendered `trviews_{index}`, producing `deblur_{index}`.
- NAFNet weights (`NAFNet-width64_k.yml`) change per iteration (up to 4 models).
- On the final iteration (`index = 5`), deblurring is skipped and `tsviews_5` is considered the final output.

Final results are saved in:
```
data/<scene_name>/Final_results/
```

Example:
```
data/cozyroom/Final_results/
```

**Note:** The radiance field module used in each iteration is modular and can be replaced with other RF representations (e.g., voxel grids, NeRF). 

## Acknowledgements

This project builds upon the following works. We thank the authors for open-sourcing their excellent codebases:

- [NAFNet (ECCV 2022)](https://github.com/megvii-research/NAFNet)
- [Plenoxels / SVoX2 (CVPR 2022)](https://github.com/sxyu/svox2)
- [3D Gaussian Splatting (SIGGRAPH 2023)](https://github.com/graphdeco-inria/gaussian-splatting)
- [LLFF: Local Light Field Fusion (SIGGRAPH 2019)](https://github.com/Fyusion/LLFF)

## Citation

If you find this work useful for your research, please consider citing:

```bibtex
@article{choi2025exploiting,
  title={Exploiting Deblurring Networks for Radiance Fields},
  author={Choi, Haeyun and Yang, Heemin and Han, Janghyeok and Cho, Sunghyun},
  journal={CVPR},
  year={2025}
}
```
