# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_lmdb_sat,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop, paired_random_crop_sat
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, padding_sat
import scipy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

iso2shot = lambda x: 9.2857e-07 * x + 8.1006e-05  # ISO => Shot noise
logshot2logread = lambda x: 2.2282 * x + 0.45982  # log(shot noise) => log(read noise)

def masks_CFA_Bayer(shape, pattern='RGGB'):
    """
    Returns the *Bayer* CFA red, green and blue masks for given pattern.

    Parameters
    ----------
    shape : array_like
        Dimensions of the *Bayer* CFA.
    pattern : unicode, optional
        **{'RGGB', 'BGGR', 'GRBG', 'GBRG'}**,
        Arrangement of the colour filters on the pixel array.

    Returns
    -------
    tuple
        *Bayer* CFA red, green and blue masks.

    Examples
    --------
    >>> from pprint import pprint
    >>> shape = (3, 3)
    >>> pprint(masks_CFA_Bayer(shape))
    (array([[ True, False,  True],
           [False, False, False],
           [ True, False,  True]], dtype=bool),
     array([[False,  True, False],
           [ True, False,  True],
           [False,  True, False]], dtype=bool),
     array([[False, False, False],
           [False,  True, False],
           [False, False, False]], dtype=bool))
    >>> pprint(masks_CFA_Bayer(shape, 'BGGR'))
    (array([[False, False, False],
           [False,  True, False],
           [False, False, False]], dtype=bool),
     array([[False,  True, False],
           [ True, False,  True],
           [False,  True, False]], dtype=bool),
     array([[ True, False,  True],
           [False, False, False],
           [ True, False,  True]], dtype=bool))
    """

    pattern = pattern.upper()

    channels = dict((channel, np.zeros(shape)) for channel in 'RGB')
    for channel, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        channels[channel][y::2, x::2] = 1

    return tuple(channels[c].astype(bool) for c in 'RGB')


class Demosaic(nn.Module):
    # based on https://github.com/GuoShi28/CBDNet/blob/master/SomeISP_operator_python/Demosaicing_malvar2004.py

    def __init__(self):
        super(Demosaic, self).__init__()

        GR_GB = np.asarray(
            [[0, 0, -1, 0, 0],
             [0, 0, 2, 0, 0],
             [-1, 2, 4, 2, -1],
             [0, 0, 2, 0, 0],
             [0, 0, -1, 0, 0]]) / 8  # yapf: disable

        # [5,5] => rot90 => [1, 1, 5, 5]
        self.GR_GB_pt = torch.tensor(np.rot90(GR_GB, k=2).copy(), dtype=torch.float32)

        Rg_RB_Bg_BR = np.asarray(
            [[0, 0, 0.5, 0, 0],
             [0, -1, 0, -1, 0],
             [-1, 4, 5, 4, - 1],
             [0, -1, 0, -1, 0],
             [0, 0, 0.5, 0, 0]]) / 8  # yapf: disable
        self.Rg_RB_Bg_BR_pt = torch.tensor(np.rot90(Rg_RB_Bg_BR, k=2).copy(), dtype=torch.float32)

        Rg_BR_Bg_RB = np.transpose(Rg_RB_Bg_BR)
        self.Rg_BR_Bg_RB_pt = torch.tensor(np.rot90(Rg_BR_Bg_RB, k=2).copy(), dtype=torch.float32)

        Rb_BB_Br_RR = np.asarray(
            [[0, 0, -1.5, 0, 0],
             [0, 2, 0, 2, 0],
             [-1.5, 0, 6, 0, -1.5],
             [0, 2, 0, 2, 0],
             [0, 0, -1.5, 0, 0]]) / 8  # yapf: disable

        self.Rb_BB_Br_RR_pt = torch.tensor(np.rot90(Rb_BB_Br_RR, k=2).copy(), dtype=torch.float32)


    def cuda(self, device=None):
        self.GR_GB_pt = self.GR_GB_pt.cuda(device)
        self.Rg_RB_Bg_BR_pt = self.Rg_RB_Bg_BR_pt.cuda(device)
        self.Rg_BR_Bg_RB_pt = self.Rg_BR_Bg_RB_pt.cuda(device)
        self.Rb_BB_Br_RR_pt = self.Rb_BB_Br_RR_pt.cuda(device)


    def forward(self, CFA_inputs, pattern='RGGB'):
        batch_size, c, h, w = CFA_inputs.shape

        R_m, G_m, B_m = masks_CFA_Bayer([h, w], pattern)

        # CFA mask
        R_m_pt = torch.from_numpy(R_m[np.newaxis, np.newaxis, :, :]).to(CFA_inputs.device)
        G_m_pt = torch.from_numpy(G_m[np.newaxis, np.newaxis, :, :]).to(CFA_inputs.device)
        B_m_pt = torch.from_numpy(B_m[np.newaxis, np.newaxis, :, :]).to(CFA_inputs.device)

        R = CFA_inputs * R_m_pt
        G = CFA_inputs * G_m_pt
        B = CFA_inputs * B_m_pt

        # True : GR_GB, False : G
        GR_GB_result = F.conv2d(CFA_inputs, weight=self.GR_GB_pt.expand(1, 1, -1, -1), padding=2, groups=1)
        Rm_Bm = np.logical_or(R_m, B_m)[np.newaxis, np.newaxis, :, :]
        Rm_Bm = np.tile(Rm_Bm, [batch_size, 1, 1, 1])
        Rm_Bm_pt = torch.tensor(Rm_Bm.copy(), dtype=torch.bool).to(CFA_inputs.device)
        G = GR_GB_result * Rm_Bm_pt + G * torch.logical_not(Rm_Bm_pt)

        RBg_RBBR = F.conv2d(CFA_inputs, weight=self.Rg_RB_Bg_BR_pt.expand(1, 1, -1, -1), padding=2, groups=1)
        RBg_BRRB = F.conv2d(CFA_inputs, weight=self.Rg_BR_Bg_RB_pt.expand(1, 1, -1, -1), padding=2, groups=1)
        RBgr_BBRR = F.conv2d(CFA_inputs, weight=self.Rb_BB_Br_RR_pt.expand(1, 1, -1, -1), padding=2, groups=1)

        # Red rows.
        R_r = np.transpose(np.any(R_m == 1, axis=1)[np.newaxis]) * np.ones(R_m.shape)
        # Red columns.
        R_c = np.any(R_m == 1, axis=0)[np.newaxis] * np.ones(R_m.shape)
        # Blue rows.
        B_r = np.transpose(np.any(B_m == 1, axis=1)[np.newaxis]) * np.ones(B_m.shape)
        # Blue columns
        B_c = np.any(B_m == 1, axis=0)[np.newaxis] * np.ones(B_m.shape)

        # rg1g2b
        Rr_Bc = R_r * B_c
        Br_Rc = B_r * R_c

        Rr_Bc = np.tile(Rr_Bc[np.newaxis, np.newaxis, :, :], [batch_size, 1, 1, 1])
        Br_Rc = np.tile(Br_Rc[np.newaxis, np.newaxis, :, :], [batch_size, 1, 1, 1])
        Rr_Bc_pt = torch.tensor(Rr_Bc.copy(), dtype=torch.bool).to(CFA_inputs.device)
        Br_Rc_pt = torch.tensor(Br_Rc.copy(), dtype=torch.bool).to(CFA_inputs.device)

        R = RBg_RBBR * Rr_Bc_pt + R * torch.logical_not(Rr_Bc_pt)
        R = RBg_BRRB * Br_Rc_pt + R * torch.logical_not(Br_Rc_pt)

        Br_Rc = B_r * R_c
        Rr_Bc = R_r * B_c

        Br_Rc = np.tile(Br_Rc[np.newaxis, np.newaxis, :, :], [batch_size, 1, 1, 1])
        Rr_Bc = np.tile(Rr_Bc[np.newaxis, np.newaxis, :, :], [batch_size, 1, 1, 1])
        Br_Rc_pt = torch.tensor(Br_Rc.copy(), dtype=torch.bool).to(CFA_inputs.device)
        Rr_Bc_pt = torch.tensor(Rr_Bc.copy(), dtype=torch.bool).to(CFA_inputs.device)

        B = RBg_RBBR * Br_Rc_pt + B * torch.logical_not(Br_Rc_pt)
        B = RBg_BRRB * Rr_Bc_pt + B * torch.logical_not(Rr_Bc_pt)

        Br_Bc = B_r * B_c
        Rr_Rc = R_r * R_c

        Br_Bc = np.tile(Br_Bc[np.newaxis, np.newaxis, :, :], [batch_size, 1, 1, 1])
        Rr_Rc = np.tile(Rr_Rc[np.newaxis, np.newaxis, :, :], [batch_size, 1, 1, 1])
        Br_Bc_pt = torch.tensor(Br_Bc.copy(), dtype=torch.bool).to(CFA_inputs.device)
        Rr_Rc_pt = torch.tensor(Rr_Rc.copy(), dtype=torch.bool).to(CFA_inputs.device)

        R = RBgr_BBRR * Br_Bc_pt + R * torch.logical_not(Br_Bc_pt)
        B = RBgr_BBRR * Rr_Rc_pt + B * torch.logical_not(Rr_Rc_pt)

        new_out = torch.cat([R, G, B], dim=1)

        return new_out


class PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder, self.lq2_folder = opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_lq2']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder, self.lq2_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt', 'lq2']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder, self.lq2_folder], ['lq', 'gt', 'lq2'])
        else:
            # 230913
            raise ValueError('Only support lmdb data backend now.')


        # elif 'meta_info_file' in self.opt and self.opt[
        #         'meta_info_file'] is not None:
        #     self.paths = paired_paths_from_meta_info_file(
        #         [self.lq_folder, self.gt_folder], ['lq', 'gt'],
        #         self.opt['meta_info_file'], self.filename_tmpl)
        # else:
        #     self.paths = paired_paths_from_folder(
        #         [self.lq_folder, self.gt_folder], ['lq', 'gt'],
        #         self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        # print('gt path,', gt_path)
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        # print(', lq path', lq_path)
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))
        
        lq2_path = self.paths[index]['lq2_path']
        # print(', lq2 path', lq2_path)
        img_bytes = self.file_client.get(lq2_path, 'lq2')
        try:
            img_lq2 = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq2 path {} not working".format(lq2_path))


        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq, img_lq2 = padding(img_gt, img_lq, img_lq2, gt_size)

            # random crop
            img_gt, img_lq, img_lq2 = paired_random_crop(img_gt, img_lq, img_lq2, gt_size, scale,
                                                gt_path)
            # flip, rotation
            img_gt, img_lq, img_lq2 = augment([img_gt, img_lq, img_lq2], self.opt['use_flip'],
                                     self.opt['use_rot'])

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq, img_lq2 = img2tensor([img_gt, img_lq, img_lq2],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            normalize(img_lq2, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq2': img_lq2,
            'lq_path': lq_path,
            'gt_path': gt_path,
            'lq2_path': lq2_path
        }

    def __len__(self):
        return len(self.paths)


class PairedImageDatasetSat(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDatasetSat, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder, self.sat_folder, self.lq2_folder = opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_sat'], opt['dataroot_lq2']

        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder, self.sat_folder, self.lq2_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt', 'sat', 'lq2']
            self.paths = paired_paths_from_lmdb_sat(
                [self.lq_folder, self.gt_folder, self.sat_folder, self.lq2_folder], ['lq', 'gt', 'sat', 'lq2'])
        
        else:
            # 240425
            raise ValueError('Only support lmdb data backend now.')
        
        # elif 'meta_info_file' in self.opt and self.opt[
        #         'meta_info_file'] is not None:
        #     self.paths = paired_paths_from_meta_info_file(
        #         [self.lq_folder, self.gt_folder], ['lq', 'gt'],
        #         self.opt['meta_info_file'], self.filename_tmpl)
        # else:
        #     self.paths = paired_paths_from_folder(
        #         [self.lq_folder, self.gt_folder], ['lq', 'gt'],
        #         self.filename_tmpl)
        
        self.ccm = torch.from_numpy(scipy.io.loadmat('C:/Users/owner/Desktop/chy/Research/231026/NAFNet/basicsr/data/CCM_matrix.mat')['colorCorrectionMatrix']).float()
        self.lin2xyz = torch.from_numpy(scipy.io.loadmat('C:/Users/owner/Desktop/chy/Research/231026/NAFNet/basicsr/data/M_lin2xyz.mat')['M']).float()
        self.xyz2lin = torch.from_numpy(scipy.io.loadmat('C:/Users/owner/Desktop/chy/Research/231026/NAFNet/basicsr/data/M_xyz2lin.mat')['M']).float()
        
        self.demosaicking_process = Demosaic()

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt, lq and sat mask images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        # print('gt path,', gt_path)
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        sat_path = self.paths[index]['sat_path']
        img_bytes = self.file_client.get(sat_path, 'sat')
        try:
            img_sat = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("sat path {} not working".format(sat_path))
        
        lq2_path = self.paths[index]['lq2_path']
        img_bytes = self.file_client.get(lq2_path, 'lq2')
        try:
            img_lq2 = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq2 path {} not working".format(lq2_path))        

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq, img_sat, img_lq2 = padding_sat(img_gt, img_lq, img_sat, img_lq2, gt_size)

            # random crop
            img_gt, img_lq, img_sat, img_lq2 = paired_random_crop_sat(img_gt, img_lq, img_sat, img_lq2, gt_size, scale,
                                                gt_path)
            # flip, rotation
            img_gt, img_lq, img_sat, img_lq2 = augment([img_gt, img_lq, img_sat, img_lq2], self.opt['use_flip'],
                                     self.opt['use_rot'])

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq, img_sat, img_lq2 = img2tensor([img_gt, img_lq, img_sat, img_lq2],
                                    bgr2rgb=True,
                                    float32=True)
        
        img_lq = img_lq.permute(1, 2, 0)
        img_sat = img_sat.permute(1, 2, 0)
        # RSBlur pipeline
        # Step 0: Parameter sampling
        alpha = np.random.uniform(low=0.25, high=1.75)  # Parameter for saturation mask scaling
        iso = np.random.uniform(low=100, high=1600)  # Random ISO sampling
        shot_noise = iso2shot(iso) + np.random.normal(loc=0., scale=5e-05)
        while shot_noise <= 0:  # Make sure that shot noise is not a negative value
            shot_noise = iso2shot(iso) + np.random.normal(loc=0., scale=5e-05)
        log_shot_noise = np.log(shot_noise)
        log_read_noise = logshot2logread(log_shot_noise) + np.random.normal(loc=0., scale=0.25)
        read_noise = np.exp(log_read_noise)
        while read_noise <= 0:
            log_read_noise = logshot2logread(log_shot_noise) + np.random.normal(loc=0., scale=0.25)
            read_noise = np.exp(log_read_noise)
        
        # Step 1: Saturation synthesis
        img_lq += (alpha * img_sat)
        img_lq = torch.clip(img_lq, min=0., max=1.)

        # Step 2: Lin2XYZ
        img_lq = torch.matmul(img_lq, self.lin2xyz)

        # Step 3: Inverse CCM
        img_lq = torch.matmul(img_lq, torch.linalg.inv(self.ccm))
        img_lq = torch.clip(img_lq, min=0., max=1.)

        # Step 4: Mosaic (RGGB)
        img_lq_mosaic = torch.zeros_like(img_lq)[:, :, 0]
        img_lq_mosaic[0::2, 0::2] = img_lq[0::2, 0::2, 0]
        img_lq_mosaic[0::2, 1::2] = img_lq[0::2, 1::2, 1]
        img_lq_mosaic[1::2, 0::2] = img_lq[1::2, 0::2, 1]
        img_lq_mosaic[1::2, 1::2] = img_lq[1::2, 1::2, 2]

        # Step 5: Noise synthesis
        img_lq_mosaic = (torch.poisson(img_lq_mosaic / shot_noise) * shot_noise)  # Shot noise synthesis
        img_lq_mosaic += torch.zeros(img_lq_mosaic.shape).normal_(mean=0, std=np.sqrt(read_noise))  # Read noise synthesis

        # Step 6: Demosaic
        img_lq = self.demosaicking_process(img_lq_mosaic.unsqueeze(0).unsqueeze(0))[0].permute(1, 2, 0)
        img_lq = torch.clip(img_lq, min=0., max=1.)

        # Step 7: CCM
        img_lq = torch.matmul(img_lq, self.ccm)

        # Step 8: XYZ2Lin
        img_lq = torch.matmul(img_lq, self.xyz2lin).permute(2, 0, 1)
        img_lq = torch.clip(img_lq, min=0., max=1.)
        
        # Step 9: Gamma Correction
        img_lq = torch.pow(img_lq, 1.0/2.2)
        img_gt = torch.pow(img_gt, 1.0/2.2)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            normalize(img_lq2, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq2': img_lq2,
            'lq_path': lq_path,
            'gt_path': gt_path,
            'lq2_path': lq2_path
        }

    def __len__(self):
        return len(self.paths)
