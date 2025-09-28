import os, sys
sys.path.append(os.getcwd())

import numpy as np
import torch

def ground_depth(intrinsics, extrinsics, cam_height, im_h, im_w, downsample= 1, trans= None, U= None, V= None):
    """
        Converts intrinsics, extrinsics and cam_height to ground depth
        Reference:
        GEDepth: Ground Embedding for Monocular Depth Estimation,
        Yang et al, ICCV 2023
        https://arxiv.org/pdf/2309.09975
        See Equation (5)

        Inputs:
            intrinsics  = B x 3 x 3
            extrinsics  = B x 4 x 4
            cam_height  = B
            downsample  = integer
            trans       = B x 3 x 3 transformation of image (crop / resize)
            U           = B x m x n (at downsampled resolution)
            V           = B x m x n (at downsampled resolution)
            im_h and im_w are for constraining the bottom pixel resolution.

        Output:
            gdepth      = B x H x W or B x m x n
    """
    im_h //= downsample
    im_w //= downsample
    if trans is None:
        trans = torch.ones_like(intrinsics)

    dsamp_matrix          = torch.zeros_like(intrinsics)
    dsamp_matrix[:, 0, 0] = 1.0/downsample
    dsamp_matrix[:, 1, 1] = 1.0/downsample
    dsamp_matrix[:, 2, 2] = 1.0
    intrinsics_final      = dsamp_matrix @ trans @ intrinsics

    rot   = extrinsics[:, :3, :3]                # B x 3 x 3
    trans = extrinsics[:, :3, 3:]                # B x 3 x 1

    rot_inv = torch.linalg.inv(rot)
    intrinsics_inv = torch.linalg.inv(intrinsics_final)
    A     = torch.bmm(rot_inv, intrinsics_inv)  # B x 3 x 3
    B     = torch.bmm(rot_inv, -trans)          # B x 3 x 1

    b2    = B[:, 1, 0]                          # B
    a21   = A[:, 1, 0]                          # B
    a22   = A[:, 1, 1]                          # B
    a23   = A[:, 1, 2]                          # B

    if U is None or V is None:
        V, U = torch.meshgrid(torch.linspace(0, im_h - 1, im_h), torch.linspace(0, im_w - 1, im_w))
        U = U.to(extrinsics.device).unsqueeze(0)    # 1 x H x W
        V = V.to(extrinsics.device).unsqueeze(0)    # 1 x H x W

    Nr = (cam_height - b2).unsqueeze(1).unsqueeze(1)
    Dr = a21.unsqueeze(1).unsqueeze(1) * U  +  a22.unsqueeze(1).unsqueeze(1) * V +  a23.unsqueeze(1).unsqueeze(1)
    gdepth = Nr/Dr

    return gdepth

def inverse_ground(intrinsics, extrinsics, cam_height, im_h, im_w, downsample= 1, trans= None, U= None, gdepth= None):
    im_h //= downsample
    im_w //= downsample
    if trans is None:
        trans = torch.ones_like(intrinsics)

    dsamp_matrix          = torch.zeros_like(intrinsics)
    dsamp_matrix[:, 0, 0] = 1.0/downsample
    dsamp_matrix[:, 1, 1] = 1.0/downsample
    dsamp_matrix[:, 2, 2] = 1.0
    intrinsics_final      = dsamp_matrix @ trans @ intrinsics

    rot   = extrinsics[:, :3, :3]                # B x 3 x 3
    trans = extrinsics[:, :3, 3:]                # B x 3 x 1

    rot_inv = torch.linalg.inv(rot)
    intrinsics_inv = torch.linalg.inv(intrinsics_final)
    A     = torch.bmm(rot_inv, intrinsics_inv)  # B x 3 x 3
    B     = torch.bmm(rot_inv, -trans)          # B x 3 x 1

    b2    = B[:, 1, 0]                          # B
    a21   = A[:, 1, 0]                          # B
    a22   = A[:, 1, 1]                          # B
    a23   = A[:, 1, 2]                          # B

    Nr = (cam_height - b2).unsqueeze(1).unsqueeze(1)
    Dr = Nr / gdepth

    V = intrinsics_final[:, 0, 0].unsqueeze(1).unsqueeze(1) *Dr + intrinsics_final[:, 1, 2].unsqueeze(1).unsqueeze(1)
    # V = Dr - a21.unsqueeze(1).unsqueeze(1) * U -  a23.unsqueeze(1).unsqueeze(1)
    # V /= a22.unsqueeze(1).unsqueeze(1)

    return V

def get_extrinsics(calibs, intrinsics):
    """
    Args:
        calibs:     B x 3 x 4
        intrinsics: B x 3 x 3

    Returns:
        extrinsics: B x 4 x 4
    """
    BATCH_SIZE = intrinsics.shape[0]
    ones = torch.from_numpy(np.array( [[0, 0., 0, 1.]]) ).unsqueeze(0).repeat(BATCH_SIZE, 1, 1).type(intrinsics.dtype).to(intrinsics.device)  # B x 1 x 4
    calibs = torch.concat((calibs, ones), dim= 1) # B x 4 x 4

    int_4  = torch.zeros_like(calibs)
    int_4[:,:3, :3] = intrinsics
    int_4[:, 3, 3]   = 1.0

    extrinsics = torch.linalg.inv(int_4) @ calibs

    return extrinsics