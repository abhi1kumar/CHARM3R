"""
    Sample Run:
    python .py
"""
import os, sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn

np.set_printoptions   (precision= 2, suppress= True)
torch.set_printoptions(precision= 2, sci_mode= False)

from lib.helpers.util import backproject_2d_pixels_in_4D_format
from lib.datasets.kitti_utils import affine_transform

def cast_to_cpu_cuda_tensor(input, reference_tensor):
    if reference_tensor.is_cuda and not input.is_cuda:
        input = input.cuda()
    if not reference_tensor.is_cuda and input.is_cuda:
        input = input.cpu()
    return input

def trans_to_crop_params(trans, K, im_width_height= (512, 512), debug= False):
    if type(trans) == np.ndarray:
        old_center = K[:2, 2].T
        cc_x, cc_y = affine_transform(pt= old_center, t= trans)
        scale = 1
        cc_x /= im_width_height[0]
        cc_y /= im_width_height[1]
        width = 1.0/trans[0, 0]
        height = 1.0/trans[1, 1]
        output = torch.tensor([cc_x, cc_y, width, height, scale])
        if debug:
            print(old_center)
            print(cc_x, cc_y)
            print(output)
    elif type(trans) == torch.Tensor:
        old_center = K[:, :2, 2].unsqueeze(2)                        # N x 2 x 1
        ONES       = torch.ones_like(old_center[:, 0].unsqueeze(1))  # N x 1 x 1
        old_center = torch.concat([old_center, ONES], dim= 1).type(trans.dtype)   # N x 3 x 1
        trans      = cast_to_cpu_cuda_tensor(trans, reference_tensor= old_center) # N x 2 x 3
        center     = torch.bmm(trans, old_center)                    # N x 2 x 1
        cc_x       = center[:, 0]/im_width_height[0]
        cc_y       = center[:, 1]/im_width_height[1]
        width      = 1.0/trans[:, 0, 0].unsqueeze(1)
        height     = 1.0/trans[:, 1, 1].unsqueeze(1)
        scale      = torch.ones_like(cc_x)
        output     = torch.concat([cc_x, cc_y, width, height, scale], dim= 1)
    return output


def compute_ndc_coordinates(
    crop_parameters=None,
    use_half_pix=True,
    num_patches_x=512,
    num_patches_y=512,
    device=None,
):
    """
    Computes NDC Grid using crop_parameters. If crop_parameters is not provided,
    then it assumes that the crop is the entire image (corresponding to an NDC grid
    where top left corner is (0, 0) and bottom right corner is (1, 1)).
    Based on Cameras as Rays, Zhang et al, ICLR 2024
    Their and our NDC is different. Their NDC is (-1, -1) and (1,1), while ours is (0, 0) and (1, 1)
    Reference: https://github.com/jasonyzhang/RayDiffusion/blob/8df11fed954066dc849255dc72420f0ce3b42f49/ray_diffusion/utils/rays.py#L494-L545
    """
    if crop_parameters is None:
        cc_x, cc_y, width, height = 0.5, 0.5, 1, 1
    else:
        if len(crop_parameters.shape) > 1:
            return torch.stack(
                [
                    compute_ndc_coordinates(
                        crop_parameters=crop_param,
                        use_half_pix=use_half_pix,
                        num_patches_x=num_patches_x,
                        num_patches_y=num_patches_y,
                    )
                    for crop_param in crop_parameters
                ],
                dim=0,
            )
        device = crop_parameters.device
        cc_x, cc_y, width, height, _ = crop_parameters

    dx = 1 / num_patches_x
    dy = 1 / num_patches_y
    if use_half_pix:
        min_y = 0.5*dy
        max_y = (num_patches_y-0.5) * dy
        min_x = 0.5*dx
        max_x = (num_patches_x-0.5) * dx
    else:
        min_y = min_x = 0
        max_y = (num_patches_y-1) * dy
        max_x = (num_patches_x-1) * dx

    y, x = torch.meshgrid(
        torch.linspace(min_y, max_y, num_patches_y, dtype=torch.float32, device=device),
        torch.linspace(min_x, max_x, num_patches_x, dtype=torch.float32, device=device),
        indexing="ij",
    )
    x_prime = (x - cc_x) * width
    y_prime = (y - cc_y) * height
    xyd_grid = torch.stack([x_prime, y_prime, torch.ones_like(x_prime)], dim=-1) # H x W x 3
    return xyd_grid


def cameras_to_rays(
    cameras,
    intrinsics,
    crop_parameters=None,
    return_mode='plucker',
    use_half_pix=True,
    num_patches_x=512,
    num_patches_y=512,
    img_width=512,
    img_height=512,
):
    """
    Unprojects rays from camera center to grid on image plane.
    Based on Cameras as Rays, Zhang et al, ICLR 2024
    Reference: https://github.com/jasonyzhang/RayDiffusion/blob/8df11fed954066dc849255dc72420f0ce3b42f49/ray_diffusion/utils/rays.py#L243-L293

    Args:
        cameras: (B, 4, 4)
        intrinsics : (B, 3, 3)
        crop_parameters: Crop parameters in NDC (cc_x, cc_y, crop_width, crop_height, scale).
            Shape is (B, 5).
        use_half_pix: If True, use half pixel offset (Default: True).
        return_mode: directions/plucker/directions_plucker (Default: 'both').
        num_patches_x: Number of patches in x direction (Default: 512).
        num_patches_y: Number of patches in y direction (Default: 512).
    """
    unprojected = []
    origins     = []
    crop_parameters_list = (
        crop_parameters if crop_parameters is not None else [None for _ in cameras]
    )
    for camera, intrinsic, crop_param in zip(cameras, intrinsics, crop_parameters_list):
        uvd_grid = compute_ndc_coordinates(
            crop_parameters=crop_param,
            use_half_pix=use_half_pix,
            num_patches_x=num_patches_x,
            num_patches_y=num_patches_y,
            device= cameras.device
        )

        # Transform from normalized coordinates to usual ones.
        uvd_grid[:, :, 0] *= img_width
        uvd_grid[:, :, 1] *= img_height
        unprojected.append(
            backproject_2d_pixels_in_4D_format(p2_inv= torch.linalg.inv(camera), points= uvd_grid.reshape(-1, 3).transpose(0, 1), pad_ones= True).transpose(0, 1)[:, :3]
        )
        optical_center = intrinsic[:, 2].unsqueeze(1)
        origins.append(
            backproject_2d_pixels_in_4D_format(p2_inv= torch.linalg.inv(camera), points= optical_center, pad_ones= True).transpose(0, 1)[:, :3]   # 1 x 3
        )
    unprojected    = torch.stack(unprojected, dim=0)                      # (N, P, 3)
    origins        = torch.stack(origins, dim= 0)                         # (N, 1, 3)
    ray_origins    = origins.repeat(1, num_patches_y * num_patches_x, 1)  # (N, P, 3)
    ray_directions = unprojected - ray_origins
    # Normalize ray directions to unit vectors
    ray_directions = ray_directions / ray_directions.norm(dim=-1, keepdim=True)
    plucker_normal = torch.cross(ray_origins, ray_directions, dim=-1)     # (N, P, 3)
    ray_directions = ray_directions.reshape(-1, num_patches_y, num_patches_x, 3).permute(0, 3, 1, 2)  # (N, 3, H, W)
    plucker_normal = plucker_normal.reshape(-1, num_patches_y, num_patches_x, 3).permute(0, 3, 1, 2)  # (N, 3, H, W)
    if return_mode == "directions":
        return ray_directions
    elif return_mode == "plucker":
        return plucker_normal
    elif return_mode == "directions_plucker":
        return torch.cat([ray_directions, plucker_normal], dim= 1)       # (N, 6, H, W)
    else:
        raise NotImplementedError
