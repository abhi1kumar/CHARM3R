import os, sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn.functional as F
import copy
import warnings
from typing import Optional

from skimage import *
# from skimage import transform
from scipy.interpolate import griddata
from lib.helpers.math_3d import backproject_2d_pixels_in_4D_format, project_3d_points_in_4D_format
from kornia.geometry.homography import find_homography_dlt, find_homography_dlt_iterated
from kornia.geometry.epipolar import normalize_points
from kornia.utils import _extract_device_dtype

EPS = 0.01

def projective_transform(mat, in_grid):
    """
        Projective/Homography transform on input_grid or pts
        mat          = 3 x 3 (np.array)      or b x p x 3 x 3 (torch.tensor) or 3 x 3 (torch.tensor)
        in_grid      = N x 2 (np.array)      or b x p x N x 2 (torch.tensor) or N x 2 (torch.tensor)
        out_grid     = N x 2 (np.array)      or b x p x N x 2 (torch.tensor)
    """
    EPS = 1e-3
    if type(mat) == np.ndarray:
        in_grid = in_grid.astype(np.float64)
        N, D = in_grid.shape
        if D == 2:
            in_grid = np.concatenate((in_grid, np.ones((N, 1))), axis= 1)

        out_grid = mat @ in_grid.transpose() # 3 X N
        out_grid[:2, :] /= (out_grid[2, :][np.newaxis, :] + EPS)
        out_grid = out_grid.transpose()      # N x 3

        return out_grid[:, :2]

    elif type(mat) == torch.Tensor:
        in_grid_dim = in_grid.dim()
        if in_grid_dim == 2:
            in_grid = in_grid[None, :, :]
        elif in_grid_dim == 4:
            _, _, N, D = in_grid.shape
            in_grid = in_grid.reshape(-1, N, D)

        q, N, D = in_grid.shape
        if D == 2:
            ones    = torch.ones((q, N, 1), dtype= in_grid.dtype, device= in_grid.device)
            in_grid = torch.concat((in_grid, ones), dim= 2) # q x N x 3

        mat_dim = mat.dim()
        if mat_dim == 2:
            mat = mat[None, None, :, :]
        elif mat_dim == 3:
            mat = mat[None, :, :]

        b, p, _,_ = mat.shape
        if in_grid_dim == 2:
            in_grid  = in_grid.repeat(b*p, 1, 1)                                     # b*p x N x 3
        mat      = mat.reshape(-1, 3, 3)                                             # b*p x 3 x 3
        out_grid = torch.bmm(mat, in_grid.permute(0, 2, 1))                          # b*p x 3 x N
        out_grid = out_grid.permute(0, 2, 1)                                         # b*p x N x 3
        out_grid[:, :, :2] /= (out_grid[:, :, 2].unsqueeze(2).repeat(1, 1, 2) + EPS) # b*p x N x 3
        out_grid = out_grid.reshape(b, p, -1, 3)                                     # b x p x N x 3

        if mat_dim == 2:
            return out_grid[0, 0, :, :2]
        elif mat_dim == 3:
            return out_grid[0, :, :, :2]
        return out_grid[:, :, :, :2]

def warp_images(basis_images, ty, tz, f, n, o, k, m= 0., tx= 0.):
    mat           = camera_matrix(tx= tx, ty= ty, tz= tz, f= f, m= m, n= n, o= o, k= k)
    output_images = _warp_images_with_mat(basis_images, mat)

    return output_images

def _warp_images_with_mat(input_images, mat):
    """
        Warp images with homography matrices
        input_images  = b x h x w (np.array)  or b x c x p x h x w (torch.tensor)
        mat           = 3 x 3 (np.array)      or b x p x 3 x 3     (torch.tensor)
        output_images = b x h x w (np.array)  or b x c x p x h x w (torch.tensor)
    """
    if type(input_images) == np.ndarray:
        b, h, w  = input_images.shape
        center   = np.array([w / 2., h / 2.])
        X, Y     = np.meshgrid(np.linspace(0, w - 1, w), np.linspace(0, h - 1, h))
        pix_grid = np.zeros((np.prod(X.shape), 2))
        pix_grid[:, 0] = X.flatten()
        pix_grid[:, 1] = Y.flatten()

        out_grid = projective_transform(mat= mat, in_grid= pix_grid - center)
        out_grid += center
        grid_1x = out_grid[:, 0].reshape((h, w))
        grid_1y = out_grid[:, 1].reshape((h, w))
        output_images = np.zeros_like(input_images)
        for j, image in enumerate(input_images):
            # out_grid: we know the image values. We want values at the usual sample_grid
            temp = griddata(points= out_grid, values= image.flatten(), xi= (X, Y), method= 'cubic', fill_value= 0.0)
            output_images[j] = np.nan_to_num(temp)

    elif type(input_images) == torch.Tensor:
        b, c, p, h, w  = input_images.shape
        center         = torch.tensor([w / 2., h /2.])
        Y, X           = torch.meshgrid(torch.linspace(0, h - 1, h), torch.linspace(0, w - 1, w))
        pix_grid       = torch.zeros((X.shape[0]*X.shape[1], 2))
        pix_grid[:, 0] = X.flatten()
        pix_grid[:, 1] = Y.flatten()
        out_grid       = projective_transform(mat= mat, in_grid= pix_grid - center)            # b x p x N x 2
        out_grid      += center[None, None, None, :]                                           # b x p x N x 2

        # Grid sample
        out_grid    = out_grid.reshape(b*p, -1, 2)                                             # b*p x N x 2
        X_new       = out_grid[:, :, 0].reshape(b*p, h, w)
        Y_new       = out_grid[:, :, 1].reshape(b*p, h, w)
        sample_grid = torch.cat((X_new.unsqueeze(3), Y_new.unsqueeze(3)), dim=3)               # b*p x h x w x 2

        # Normalize in range [0, 1]
        sample_grid[:, :, :, 0] = sample_grid[:, :, :, 0] / w
        sample_grid[:, :, :, 1] = sample_grid[:, :, :, 1] / h
        # Normalize in range [-1, 1]
        sample_grid = 2 * sample_grid - 1

        # In the spatial (4-D) case, for input with shape (N,C,Hin,Win)
        # and grid with shape (N,Hout,Wout,2),
        # the grid_sample output will have shape (N,C,Hout,Wout)
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        input_images  = input_images.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)                # b*p x c x h x w
        output_images = F.grid_sample(input_images, sample_grid).reshape(b, p, c, h, w)         # b x p x c x h x w
        output_images = output_images.permute(0, 2, 1, 3, 4)                                    # b x c x p x h x w

    return  output_images

def _warp_images_with_per_pixel_mat(input_images, mat, transform_about_center= False):
    """
    Transforms an image tensor with per-pixel homography.

    Args:
        image: A tensor of shape (B, C, H, W) representing the image to transform.
        mat  : A tensor of shape (B, H, W, 3, 3) representing the per-pixel homography matrices from new to old image.
        This is NOT homography old to new

    Returns:
        A transformed image tensor of the same shape as the input image.
    """
    B, C, H, W = input_images.shape

    if type(input_images) == torch.Tensor:
        device     = input_images.device
        center     = torch.tensor([W / 2., H / 2.], device= device).float()
        homography = mat.view(B, H, W, 3, 3)                     # B x H x W x 3 x 3

        Y, X = torch.meshgrid(torch.arange(H, device= device).float(), torch.arange(W, device= device).float())
        grid = torch.stack((X, Y), dim=2).view(1, H, W, 2)      # 1 x H x W x 2

        # Perform perspective transformation using matrix multiplication
        centered_grid = grid
        if transform_about_center:
            centered_grid -= center
        centered_grid = centered_grid.expand(B, -1, -1, -1)     # B x H x W x 2
        warped_grid   = projective_transform(homography.reshape(B, -1, 3, 3), in_grid= centered_grid.reshape(B, -1, 1, 2))  # B x P x 1 x 2
        if transform_about_center:
            warped_grid  += center[None, None, None, :]                                       # B x P x 1 x 2
        warped_grid   = warped_grid.reshape(B, H, W, 2)                                       # B x H x W x 2

        # Normalize in range [0, 1]
        warped_grid[:, :, :, 0] /= W
        warped_grid[:, :, :, 1] /= H
        # Normalize in range [-1, 1]
        sample_grid = 2 * warped_grid - 1

        # In the spatial (4-D) case, for input with shape (N,C,Hin,Win)
        # and grid with shape (N,Hout,Wout,2),
        # the grid_sample output will have shape (N,C,Hout,Wout)
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        output_images = F.grid_sample(input_images, sample_grid, mode= "nearest")

    return output_images

def homography_on_points(hom_matrix, src):
    """
    Homography transform of 2d points appened with ones to 2d using projection matrix
    :param hom_matrix: np array 3 x 3
    :param src:        np array N x 2
    :return out:       np array N x 2
    Deprecated in favour of projective_transform
    """
    EPS = 1e-3
    if src.ndim == 1:
        src = np.append(src, 1.)[:, None]
        out   = hom_matrix @ src
        out[:2, 0] /= (out[2, 0] + EPS)
        return out[:2, 0]
    elif src.ndim == 2:
        num_pts = src.shape[0]
        pts = np.ones((num_pts, 3, 1))
        pts[:, :2, 0] = src
        out = hom_matrix[np.newaxis, :, :] @ pts
        out[:, :2, 0] /= (out[:, 2, 0][:, np.newaxis] + EPS)
        return out[:, :2, 0]

def homography_on_image(hom_matrix, image):
    h, w = image.shape
    src = np.array([[1, 1.], [1, 3], [3, 1.], [3, 3]])
    src[:, 0] *= h/4
    src[:, 1] *= w/4

    dst = np.zeros(src.shape)
    # for i, t in enumerate(src):
    #     # x = homo_mat[:2, :2] @ (t[:, None] + homo_mat[:2, 2])   - homo_mat[:2, 2]
    #     final_hom = hom_mat
    #     x = homography_transform(final_hom, t)
    #     dst[i] = x
    dst = homography_on_points(hom_matrix, src)

    tform = transform.estimate_transform('projective',
                                             src,
                                             dst)
    transformed = transform.warp(image, tform.inverse)
    scale = np.max(image)/(np.max(transformed) + EPS)
    transformed_scaled = transformed * scale

    return transformed_scaled, transformed, scale, src, dst



def generate_homography_matrices(num_pts, transform_about_center= False, h= None, w= None):
    # See Sec 4.1 of https://arxiv.org/pdf/2306.01623.pdf
    scale   = np.random.uniform(0.5, 1.5, (num_pts, 2))
    rot_deg = np.random.uniform(-20,  20,  num_pts)
    trans   = np.random.uniform(-4 ,   4, (num_pts, 2))

    scale_mat = np.zeros((num_pts, 3, 3))
    scale_mat[:, 0, 0] = scale[:, 0]
    scale_mat[:, 1, 1] = scale[:, 1]
    scale_mat[:, 2, 2] = 1.

    rot = rot_deg * np.pi / 180
    rot_mat = np.zeros((num_pts, 3, 3))
    rot_mat[:, 0, 0] = np.cos(rot)
    rot_mat[:, 0, 1] = -np.sin(rot)
    rot_mat[:, 1, 0] = np.sin(rot)
    rot_mat[:, 1, 1] = np.cos(rot)
    rot_mat[:, 2, 2] = 1.

    trans_mat = np.zeros((num_pts, 3, 3))
    trans_mat[:, 0, 0] = 1.
    trans_mat[:, 1, 1] = 1.
    trans_mat[:, 2, 2] = 1.
    trans_mat[:, 0, 2] = trans[:, 0]
    trans_mat[:, 1, 2] = trans[:, 1]

    hom_mat = scale_mat @ rot_mat @ trans_mat

    if transform_about_center:
        right_hom_mat = np.eye(3)
        right_hom_mat[0,2] = -h/2.
        right_hom_mat[1,2] = -w/2.

        left_hom_mat = np.eye(3)
        left_hom_mat[0,2] = h/2.
        left_hom_mat[1,2] = w/2.

        hom_mat = left_hom_mat[None, :, :] @ hom_mat @ right_hom_mat[None, :, :]

    return hom_mat

def homography_on_points(hom_matrix, src):
    if src.ndim == 1:
        src = np.append(src, 1.)[:, None]
        out   = hom_matrix @ src
        return out[:2, 0]
    elif src.ndim == 2:
        num_pts = src.shape[0]
        pts = np.ones((num_pts, 3, 1))
        pts[:, :2, 0] = src
        out = hom_matrix[np.newaxis, :, :] @ pts
        return out[:, :2, 0]

def homography_on_image(hom_matrix, image):
    h, w = image.shape
    src = np.array([[1, 1.], [1, 3], [3, 1.], [3, 3]])
    src[:, 0] *= h/4
    src[:, 1] *= w/4

    dst = np.zeros(src.shape)
    # for i, t in enumerate(src):
    #     # x = homo_mat[:2, :2] @ (t[:, None] + homo_mat[:2, 2])   - homo_mat[:2, 2]
    #     final_hom = hom_mat
    #     x = homography_transform(final_hom, t)
    #     dst[i] = x
    dst = homography_on_points(hom_matrix, src)

    tform = transform.estimate_transform('projective',
                                             src,
                                             dst)
    transformed = transform.warp(image, tform.inverse)
    scale = np.max(image)/(np.max(transformed) + EPS)
    transformed_scaled = transformed * scale

    return transformed_scaled, transformed, scale, src, dst

def z_buffered_index(vectors, depth):
    """
    Returns z-buffered index of each vector
    Args:
        vectors: tensor of shape B x P x 2
        depth: another tensor of shape B x P

    Returns:
        z_buffer_index of shape B x P
    """
    EPS = 0.5
    M   = 100
    with torch.no_grad():
        dist = torch.cdist(vectors, vectors)  # B x P x P

        # Find other (non-diagonal) points which are closest
        diag_index = torch.arange(dist.shape[1], device=dist.device)
        dist[:, diag_index, diag_index] = M  # Diagonals are not chosen

        other_dist, other_index = torch.min(dist, dim=2)    # B x P
        other_mask  = other_dist < EPS                      # B x P
        other_depth = torch.gather(depth, dim= 1, index= other_index)

        # If other points have depth less than current depth
        depth_mask  = other_depth < depth                   # B x P

        # Replace if other point exist and other point has smaller depth
        mask         = torch.logical_and(other_mask, depth_mask).float()
        buffer_index = mask * other_index + (1 - mask) * diag_index

    return buffer_index.long()

def invert_Homography(Hom_01, depth= None):
    """
    Invert Homography matrices
    Args:
        Hom_01: Homography from camera 0 to camera 1: tensor of shape B x H x W x 3 x 3
        depth:  B x H x W

    Returns:
        Hom_01: Homography from camera 1 to camera 0: tensor of shape B x H x W x 3 x 3
    """
    squeeze = False
    if Hom_01.ndim == 4:
        squeeze = True
        Hom_01  = Hom_01.unsqueeze(0)
        depth   = depth.unsqueeze(0)

    SMALL          = 0.5
    B, h, w, _, _  = Hom_01.shape
    identity       = torch.eye(3).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(Hom_01.device).repeat(B, h, w, 1, 1)
    Hom_01        += SMALL * identity
    Hom_10_vanilla = torch.linalg.inv(Hom_01.clone())  # B x H x W x 3 x 3

    if depth is not None:
        B, H, W, _, _ = Hom_01.shape
        device = Hom_01.device

        Y, X = torch.meshgrid(torch.arange(H, device=device).float(), torch.arange(W, device=device).float())
        grid = torch.stack((X, Y), dim=2).view(1, H, W, 2)  # 1 x H x W x 2

        # Perform perspective transformation using matrix multiplication
        centered_grid = grid
        centered_grid = centered_grid.expand(B, -1, -1, -1)  # B x H x W x 2
        warped_grid   = projective_transform(Hom_01.reshape(B, -1, 3, 3), in_grid=centered_grid.reshape(B, -1, 1, 2))  # B x P x 1 x 2
        warped_grid   = warped_grid.reshape(B, -1, 2)          # B x HW x 2

        buffer_index   = z_buffered_index(warped_grid, depth.reshape(B, -1))                  # B x HW
        buffer_index   = buffer_index.unsqueeze(2).repeat(1, 1, 9)                            # B x HW x 9

        Hom_10_vanilla = Hom_10_vanilla.reshape(B, -1, 9)                                     # B x HW x 9
        Hom_10_vanilla = torch.gather(Hom_10_vanilla, dim= 1, index= buffer_index)            # B x HW x 9
        Hom_10         = Hom_10_vanilla.reshape(B, H, W, 3, 3)                                # B x H x W x 3 x 3

    else:
        Hom_10         = Hom_10_vanilla

    if squeeze:
        Hom_10  = Hom_10[0]

    return Hom_10

def custom_clamp(input, EPS= 0.5):
    """
    Custom clamping function for the specified conditions.

    Args:
    input: The input tensor or array

    Returns:
    The clamped tensor or array
    """
    lower_th = -EPS
    upper_th = -EPS

    if type(input) == torch.Tensor:
        mid_lower_mask = torch.logical_and(input >= -EPS, input < 0)
        mid_upper_mask = torch.logical_and(input >=  0  , input <= EPS)
        clamped_input  = input.clone()

    elif type(input) == np.ndarray:
        mid_lower_mask = np.logical_and(input >= -EPS, input < 0)
        mid_upper_mask = np.logical_and(input >=  0  , input <= EPS)
        clamped_input  = copy.deepcopy(input)

    else:
        if (input >= -EPS) & (input < 0):
            clamped_input = lower_th
        elif (input >=  0) & (input <= EPS):
            clamped_input = upper_th
        else:
            clamped_input = input

    if type(input) == torch.Tensor or type(input) == np.ndarray:
        clamped_input[mid_lower_mask] = lower_th
        clamped_input[mid_upper_mask] = upper_th

    return clamped_input

def camera_matrix(tx= 0., ty= 0., tz= 0., f= 1., m= 0., n= 1., o= 1., k= 10., u0= 256., v0= 256., use_plane= True):
    # tx, ty, tz are in KITTI image coordinates
    k = custom_clamp(k)
    if use_plane:
        # Use homography of planes
        # Plane are of the form mx + ny + oz + k = 0
        # This is exactly as the Theorem 1, Equation 7 of the DEVIANT, ECCV 2022
        if type(m) == torch.Tensor:
            if m.ndim == 2:
                h, w = m.shape
                mat = torch.zeros((h, w, 3, 3), dtype= m.dtype, device= m.device)
                mat[:, :, 0, 0] = (1+tx*m/k)*f
                mat[:, :, 0, 1] =    tx*n/k *f
                mat[:, :, 0, 2] =    tx*o/k *f*f
                mat[:, :, 1, 0] =    ty*m/k *f
                mat[:, :, 1, 1] = (1+ty*n/k)*f
                mat[:, :, 1, 2] =    ty*o/k *f*f
                mat[:, :, 2, 0] =    tz*m/k
                mat[:, :, 2, 1] =    tz*n/k
                mat[:, :, 2, 2] = (1+tz*o/k)*f

                normalize = mat[:, :, 2, 2].unsqueeze(2).unsqueeze(3).repeat(1, 1, 3, 3)

            elif m.ndim == 3:
                b, h, w = m.shape
                mat = torch.zeros((b, h, w, 3, 3), dtype= m.dtype, device= m.device)
                mat[:, :, :, 0, 0] = (1+tx*m/k)*f
                mat[:, :, :, 0, 1] =    tx*n/k *f
                mat[:, :, :, 0, 2] =    tx*o/k *f*f
                mat[:, :, :, 1, 0] =    ty*m/k *f
                mat[:, :, :, 1, 1] = (1+ty*n/k)*f
                mat[:, :, :, 1, 2] =    ty*o/k *f*f
                mat[:, :, :, 2, 0] =    tz*m/k
                mat[:, :, :, 2, 1] =    tz*n/k
                mat[:, :, :, 2, 2] = (1+tz*o/k)*f

                normalize = mat[:, :, :, 2, 2].unsqueeze(3).unsqueeze(4).repeat(1, 1, 1, 3, 3)

        else:
            mat = np.eye(3).astype(np.float64)
            mat[0, 0] = (1+tx*m/k)*f
            mat[0, 1] =    tx*n/k *f
            mat[0, 2] =    tx*o/k *f*f
            mat[1, 0] =    ty*m/k *f
            mat[1, 1] = (1+ty*n/k)*f
            mat[1, 2] =    ty*o/k *f*f
            mat[2, 0] =    tz*m/k
            mat[2, 1] =    tz*n/k
            mat[2, 2] = (1+tz*o/k)*f

            normalize = np.repeat(np.repeat(np.array([[ mat[2,2] ]]), 3, axis= 0), 3, axis= 1)

    else:
        # Use homography of points
        if type(k) == torch.Tensor:
            if k.ndim == 2:
                h, w  = k.shape
                mat   = torch.zeros((h, w, 3, 3), dtype= k.dtype, device= k.device)
                depth = -k
                mat[:, :, 0, 0] = depth
                mat[:, :, 0, 2] = f*tx + u0*tz
                mat[:, :, 1, 1] = depth
                mat[:, :, 1, 2] = f*ty + v0*tz
                mat[:, :, 2, 2] = depth + tz

                normalize = mat[:, :, 2, 2].unsqueeze(2).unsqueeze(3).repeat(1, 1, 3, 3)

            elif k.ndim == 3:
                b, h, w = k.shape
                mat = torch.zeros((b, h, w, 3, 3), dtype=k.dtype, device=k.device)
                depth = -k
                mat[:, :, :, 0, 0] = depth
                mat[:, :, :, 0, 2] = f * tx + u0 * tz
                mat[:, :, :, 1, 1] = depth
                mat[:, :, :, 1, 2] = f * ty + v0 * tz
                mat[:, :, :, 2, 2] = depth + tz

                normalize = mat[:, :, :, 2, 2].unsqueeze(3).unsqueeze(4).repeat(1, 1, 1, 3, 3)

    # Last entry in homography matrices are ones.
    mat /= normalize

    return mat

def get_intrinsics_matrix(w, h, four_by_four= False):
    intrinsic_matrix = np.eye(3)
    # Omni3D
    # Refer https://github.com/facebookresearch/omni3d/blob/778bd0210e5e1b584cc52daeba57cae3a3ee8c4e/demo/demo.py#L66-L79
    f = 2 * h
    intrinsic_matrix[0, 0] = f
    intrinsic_matrix[1, 1] = f
    intrinsic_matrix[0, 2] = w / 2.0
    intrinsic_matrix[1, 2] = h / 2.0

    if four_by_four:
        temp = np.eye(4)
        temp[:3, :3] = intrinsic_matrix
        intrinsic_matrix = temp

    return intrinsic_matrix

def get_intrinsics_from_fov(w, h, fov_degree, cx= None, cy= None, four_by_four= False):
    intrinsic_matrix = np.eye(3)
    if cx is None:
        cx = w / 2.
    if cy is None:
        cy = h / 2.
    # See https://stackoverflow.com/a/41137160
    fov= fov_degree * np.pi/180.
    fx = 0.5 * w / np.tan(fov/2.)
    fy = 0.5 * h / np.tan(fov/2.)
    intrinsic_matrix[0, 0] = fx
    intrinsic_matrix[1, 1] = fy

    intrinsic_matrix[0, 2] = cx
    intrinsic_matrix[1, 2] = cy

    if four_by_four:
        temp = np.eye(4)
        temp[:3, :3] = intrinsic_matrix
        intrinsic_matrix = temp

    return intrinsic_matrix

def hom_to_grid_offset(Hom, kernel_grid, kernel_center, pix_grid= None):
    """
    Input:
    Hom:           tensor b x h x w x 3 x 3
    kernel_grid:   tensor             n x 2
    kernel_center: tensor                 2
    pix_grid:      tensor     h x w     x 2

    The last index in kernel_grid is in [x, y] format with ordering along X
    0 1 2
    3 4 5
    6 7 8
    and each containing x and y. In other words, kernel_grid is of format
    (0,0)
    (1,0)
    (2,0)
    (0,1)
    (1,1)
    (2,1)
    (0,2)
    (1,2)
    (2,2)

    Output:
    grid_offset: tensor b x h x w x n x 2
    """
    b, h, w, _, _ = Hom.shape
    Hom           = Hom.reshape((b, -1, 3, 3))                                 # b x p x 3 x 3
    if pix_grid is None:
        in_grid     = kernel_grid[None, None, :, :].repeat(b, h * w, 1, 1)             # b x p x n x 2
        out_grid    = projective_transform(mat= Hom, in_grid=kernel_grid - kernel_center)     # b x p x n x 2
        out_grid   += kernel_center[None, None, None, :]                               # b x p x n x 2
    else:
        pix_grid    = pix_grid[None, :, :, None, :].repeat(b, 1, 1, 1, 1)              # b x h x w x 1 x 2
        pix_grid    = pix_grid.reshape(b, -1, 1, 2)                                    # b x p x 1 x 2
        # Bring kernel grid in shape [-kh, kh] * [-kw, kw]
        kernel_grid = kernel_grid - torch.mean(kernel_grid, dim= 0)[None, :]           # n x 2
        kernel_grid = kernel_grid[None, None, :, :]                                    # 1 x 1 x n x 2
        in_grid     = pix_grid + kernel_grid                                           # b x p x n x 2
        # If you use in_grid, center is the center of the feature map or intrinsics
        center      = torch.tensor([h/2., w/2.], dtype= Hom.dtype, device= Hom.device)
        center      = center[None, None, None, :]                                      # 1 x 1 x 1 x 2
        out_grid    = projective_transform(mat= Hom, in_grid= in_grid)                 # b x p x n x 2
        # out_grid   += center                                                         # b x p x n x 2

    # out_grid[:, :, :, 1] = torch.clamp(out_grid[:, :, :, 1], 0, h)
    # out_grid[:, :, :, 0] = torch.clamp(out_grid[:, :, :, 0], 0, w)
    grid_offset   = out_grid - in_grid                                                 # b x p x n x 2
    grid_offset   = grid_offset.reshape(b, h, w, -1, 2)                                # b x h x w x n x 2

    return grid_offset

def grid_offset_to_deform_offset(offset):
    """
    Input:
    grid_offset: tensor b x h x w x k*k x 2
    The last index is in [x, y] format with ordering along X
    0 1 2
    3 4 5
    6 7 8
    and each containing x and y. In other wrods, it is of format
    (0,0)
    (1,0)
    (2,0)
    (0,1)
    (1,1)
    (2,1)
    (0,2)
    (1,2)
    (2,2)

    Output:
    deform_offset: tensor b x 2*n x h x w
    """
    b, h, w, n, _ = offset.shape
    # A deform offset is like `[y0, x0, y1, x1, y2, x2, y3, x3..., y8, x8]`.
    # The spatial arrangement is like:
    #     (x0, y0) (x1, y1) (x2, y2)
    #     (x3, y3) (x4, y4) (x5, y5)
    #     (x6, y6) (x7, y7) (x8, y8)
    # See https://mmcv.readthedocs.io/en/latest/_modules/mmcv/ops/deform_conv.html
    # Swap x and y first. Easy to do since last dimension.
    offset = offset[:, :, :, :, [1,0]]
    offset = offset.reshape(-1, n, 2)                        # b*h*w x n x 2
    offset = offset.reshape(-1, 2*n)
    offset = offset.reshape(b, h, w, 2*n).permute(0,3,1,2)   # b x 2*n x h x w

    return offset

def find_homography_regularized(
    points1: torch.Tensor, points2: torch.Tensor, weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""Compute the homography matrix using the DLT formulation.

    The linear system is solved by using the Weighted Least Squares Solution for the 4 Points algorithm.

    Args:
        points1: A set of points in the first image with a tensor shape :math:`(B, N, 2)`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2)`.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(B, N)`.

    Returns:
        the computed homography matrix with shape :math:`(B, 3, 3)`.
    """
    if points1.shape != points2.shape:
        raise AssertionError(points1.shape)
    if not (len(points1.shape) >= 1 and points1.shape[-1] == 2):
        raise AssertionError(points1.shape)
    if points1.shape[1] < 4:
        raise AssertionError(points1.shape)

    device, dtype = _extract_device_dtype([points1, points2])

    eps: float = 1e-8
    points1_norm, transform1 = normalize_points(points1)
    points2_norm, transform2 = normalize_points(points2)

    x1, y1 = torch.chunk(points1_norm, dim=-1, chunks=2)  # BxNx1
    x2, y2 = torch.chunk(points2_norm, dim=-1, chunks=2)  # BxNx1
    ones, zeros = torch.ones_like(x1), torch.zeros_like(x1)

    # DIAPO 11: https://www.uio.no/studier/emner/matnat/its/nedlagte-emner/UNIK4690/v16/forelesninger/lecture_4_3-estimating-homographies-from-feature-correspondences.pdf  # noqa: E501
    ax = torch.cat([zeros, zeros, zeros, -x1, -y1, -ones, y2 * x1, y2 * y1, y2], dim=-1)
    ay = torch.cat([x1, y1, ones, zeros, zeros, zeros, -x2 * x1, -x2 * y1, -x2], dim=-1)
    A = torch.cat((ax, ay), dim=-1).reshape(ax.shape[0], -1, ax.shape[-1])

    if weights is None:
        # All points are equally important
        A = A.transpose(-2, -1) @ A
    else:
        # We should use provided weights
        if not (len(weights.shape) == 2 and weights.shape == points1.shape[:2]):
            raise AssertionError(weights.shape)
        w_diag = torch.diag_embed(weights.unsqueeze(dim=-1).repeat(1, 1, 2).reshape(weights.shape[0], -1))
        A = A.transpose(-2, -1) @ w_diag @ A

    SMALL = 0.5
    identity = torch.eye(9).unsqueeze(0).repeat(A.shape[0], 1, 1).to(device)
    A += SMALL * identity

    identity    = torch.eye(3).unsqueeze(0).repeat(transform2.shape[0], 1, 1).to(device)
    transform2 += SMALL * identity

    try:
        _, _, V = torch.svd(A)
    except RuntimeError:
        warnings.warn('SVD did not converge', RuntimeWarning)
        return torch.empty((points1_norm.size(0), 3, 3), device=device, dtype=dtype)

    H = V[..., -1].view(-1, 3, 3)
    H = transform2.inverse() @ (H @ transform1)
    H_norm = H / custom_clamp(H[..., -1:, -1:], SMALL)

    return H_norm

def get_per_pixel_image_homo(depth, normal, tx = 0., ty= 0.76, tz = 0., fov_degree= 64.56, device = torch.device("cuda:0"), method= "infinite_plane"):
    """
    Generate per_pixel_homography matrices from new to old image.
    Output = h x w x 3 x 3
    """
    _, h, w = normal.shape

    if type(depth) == np.ndarray:
        depth  = torch.from_numpy(depth).float().to(device)   # h x w
    else:
        if depth.ndim == 3:
            depth  = depth[0]
    if type(normal) == np.ndarray:
        normal = torch.from_numpy(normal).float().to(device)  # 3 x h x w

    intrinsics = get_intrinsics_from_fov(w= w, h= h, fov_degree= fov_degree)
    f  = intrinsics[0, 0].item()
    u0 = intrinsics[0, 2].item()
    v0 = intrinsics[1, 2].item()
    p2         = torch.eye(4).float()
    p2[:3, :3] = torch.from_numpy(intrinsics).float()
    p2         = p2.to(device)

    if method == "infinite_plane":
        # use plane homography calculation
        # Theorem 1 of DEVIANT, Kumar et al, ECCV 2022
        Y, X       = torch.meshgrid(torch.linspace(0, h-1, h), torch.linspace(0, w-1, w))
        X          = X.to(device)
        Y          = Y.to(device)
        p2_inv     = torch.linalg.inv(p2)
        points     = torch.concat((X.flatten().unsqueeze(0), Y.flatten().unsqueeze(0), depth.flatten().unsqueeze(0)), dim= 0)       # 3 x hw
        points_3d  = backproject_2d_pixels_in_4D_format(p2_inv, points= points, pad_ones= True)                                     # 4 x hw
        points_3d  = points_3d[:3].transpose(0, 1)                                         # hw x 3
        K          = -torch.sum(points_3d * normal.permute(1, 2, 0).reshape(-1, 3), dim= 1) # hw
        K          = K.reshape(h, w)                                                       # h x w
        points     = points.reshape(3, h, w)                                               # 3 x h x w
        points_3d  = points_3d.permute(1, 0).reshape(3, h, w)                              # 3 x h x w
        homography = camera_matrix(tx= tx, ty= ty, tz= tz, f= f, m= normal[0], n= normal[1], o= normal[2], k= K) # h x w x 3 x 3

    elif method == "plane_sampling":
        # Sample points on per-point planes
        # Then calculcate homography from those points
        img0_sampled, points_3d = sample_points_on_plane(normal, depth, p2)  # h x w x 4 x 3, h x w x 3 x 3
        img0_sampled = img0_sampled.reshape(-1, 3)                 # hw4 x 3

        img1_sampled = copy.deepcopy(img0_sampled)
        img1_sampled[:, 0] -= tx
        img1_sampled[:, 1] -= ty
        img1_sampled[:, 2] -= tz

        pix0  = project_3d_points_in_4D_format(p2, points_4d= img0_sampled.transpose(1, 0), pad_ones= True).transpose(1, 0)  # hw4 x 3
        pix1  = project_3d_points_in_4D_format(p2, points_4d= img1_sampled.transpose(1, 0), pad_ones= True).transpose(1, 0)  # hw4 x 3

        pix0 = pix0[:, :2].reshape(-1, 4, 2)  # hw x 4 x 2
        pix1 = pix1[:, :2].reshape(-1, 4, 2)  # hw x 4 x 2

        # find_homography_dlt_iterated is more stable
        homography = find_homography_dlt_iterated(pix1, pix0, weights= torch.ones((h*w, 4)).to(device)).reshape(h, w, 3, 3)    # h x w x 3 x 3

    elif method == "points":
        # use point homography calculation
        homography = camera_matrix(tx= tx, ty= ty, tz= tz, f= f, k= -depth, u0= u0, v0= v0, use_plane= False)    # h x w x 3 x 3

    else:
        raise NotImplementedError

    return homography, intrinsics, depth, normal


def get_per_pixel_image_homo_batched(depth, normal, tx = 0., ty= 0.76, tz = 0., fov_degree= 64.56, device = torch.device("cuda:0"), method= "infinite_plane"):
    """
    Generate per_pixel_homography matrices from new to old image.
    depth  = b x h x w
    normal = b x 3 x h x w
    Output = b x h x w x 3 x 3
    """
    b, _, h, w = normal.shape

    if type(depth) == np.ndarray:
        depth  = torch.from_numpy(depth).unsqueeze(0).float().to(device)   # 1 x h x w
    else:
        if depth.ndim == 2:
            depth  = depth.unsqueeze(0)                                    # 1 x h x w

    if type(normal) == np.ndarray:
        normal = torch.from_numpy(normal).unsqueeze(0).float().to(device)  # 1 x 3 x h x w
    else:
        if normal.ndim == 3:
            normal = normal.unsqueeze(0)                                   # 1 x 3 x h x w

    intrinsics = get_intrinsics_from_fov(w= w, h= h, fov_degree= fov_degree)
    f  = intrinsics[0, 0].item()
    u0 = intrinsics[0, 2].item()
    v0 = intrinsics[1, 2].item()
    p2         = torch.eye(4).float()
    p2[:3, :3] = torch.from_numpy(intrinsics).float()
    p2         = p2.to(device)

    if method == "infinite_plane":
        # use plane homography calculation
        # Theorem 1 of DEVIANT, Kumar et al, ECCV 2022
        Y, X       = torch.meshgrid(torch.linspace(0, h-1, h), torch.linspace(0, w-1, w))
        X          = X.to(device).unsqueeze(0).repeat(b, 1, 1)
        Y          = Y.to(device).unsqueeze(0).repeat(b, 1, 1)
        p2_inv     = torch.linalg.inv(p2)
        points     = torch.concat((X.flatten().unsqueeze(0), Y.flatten().unsqueeze(0), depth.flatten().unsqueeze(0)), dim= 0)       # 3 x bhw
        points_3d  = backproject_2d_pixels_in_4D_format(p2_inv, points= points, pad_ones= True)                                     # 4 x bhw
        points_3d  = points_3d[:3].transpose(0, 1)                                         # bhw x 3
        K          = -torch.sum(points_3d * normal.permute(0, 2, 3, 1).reshape(-1, 3), dim= 1) # bhw
        K          = K.reshape(b, h, w)                                                       # h x w
        points     = points.reshape(3, b, h, w)                                               # 3 x h x w
        points_3d  = points_3d.permute(1, 0).reshape(3, b, h, w)                              # 3 x h x w
        homography = camera_matrix(tx= tx, ty= ty, tz= tz, f= f, m= normal[:, 0], n= normal[:, 1], o= normal[:, 2], k= K) # h x w x 3 x 3

    elif method == "plane_sampling":
        # Sample points on per-point planes
        # Then calculcate homography from those points
        img0_sampled, points_3d = sample_points_on_plane_batched(normal, depth, p2)  # b x h x w x 4 x 3, b x h x w x 3 x 3
        img0_sampled = img0_sampled.reshape(-1, 3)                 # bhw4 x 3

        img1_sampled = img0_sampled.clone().detach()
        img1_sampled[:, 0] = img1_sampled[:, 0] - tx
        img1_sampled[:, 1] = img1_sampled[:, 1] - ty
        img1_sampled[:, 2] = img1_sampled[:, 2] - tz

        pix0  = project_3d_points_in_4D_format(p2, points_4d= img0_sampled.transpose(1, 0), pad_ones= True).transpose(1, 0)  # bhw4 x 3
        pix1  = project_3d_points_in_4D_format(p2, points_4d= img1_sampled.transpose(1, 0), pad_ones= True).transpose(1, 0)  # bhw4 x 3

        pix0 = pix0[:, :2].reshape(-1, 4, 2)  # bhw x 4 x 2
        pix1 = pix1[:, :2].reshape(-1, 4, 2)  # bhw x 4 x 2

        # find_homography_dlt_iterated is more stable
        homography = find_homography_dlt_iterated(pix1, pix0, weights= torch.ones((b*h*w, 4)).to(device)).reshape(b, h, w, 3, 3)  # b x h x w x 3 x 3

    elif method == "points":
        # use point homography calculation
        homography = camera_matrix(tx= tx, ty= ty, tz= tz, f= f, k= -depth, u0= u0, v0= v0, use_plane= False)    # h x w x 3 x 3

    else:
        raise NotImplementedError

    return homography, intrinsics, depth, normal

def homography_g2im(cam_height, K, cam_pitch= 0):
    # Homography of ground to image
    # Reference: https://github.com/yuliangguo/3D_Lane_Synthetic_Dataset/blob/master/utils/utils.py#L178-L184
    # transform top-view region to original image region
    R_g2c = np.array([[1, 0, 0],
                      [0, np.cos(np.pi / 2 + cam_pitch), -np.sin(np.pi / 2 + cam_pitch)],
                      [0, np.sin(np.pi / 2 + cam_pitch), np.cos(np.pi / 2 + cam_pitch)]])
    H_g2im = np.matmul(K, np.concatenate([R_g2c[:, 0:2], [[0], [cam_height], [0]]], 1))

    return H_g2im

def get_extrinsics(yaw_degree, pitch_degree= 0, translation= np.zeros((3,))):
    yaw   = yaw_degree * np.pi/180.
    pit   = pitch_degree * np.pi/180.
    extrinsics    = np.eye(4)
    # yaw is clockwise about Z
    # pitch is clockwise about X
    # Refer: https://msl.cs.uiuc.edu/planning/node102.html where rotations are anti-clockwise
    yaw_matrix    = np.array( [[np.cos(yaw), np.sin(yaw), 0], [-np.sin(yaw), np.cos(yaw), 0], [0, 0, 1.]])  # 3 x 3
    pitch_matrix  = np.array( [[1, 0, 0], [0, np.cos(pit), -np.sin(pit)], [0, np.sin(pit), np.cos(pit)]])   # 3 x 3
    rotation_mat  = np.matmul(yaw_matrix, pitch_matrix)                                                     # 3 x 3
    extrinsics[:3, :3] = rotation_mat
    extrinsics[:3,  3] = translation

    return extrinsics

def carla_to_homo(input, camera_flag= True):
    """
    Converts Carla to Homogeneous coordinate
    :param input:
    :param camera_flag:
    :return:
    """
    temp = copy.deepcopy(input)
    if camera_flag:
        temp[[1,2], :] = temp[[2,1],:]
        temp[:, [1,2]] = temp[:, [2,1]]
    elif input.ndim == 2:
        temp[:, [1,2]] = input[:, [2,1]]
    elif input.ndim == 1:
        temp[1] = input[2]
        temp[2] = input[1]

    return temp

def get_pix_grid_center(h, w):
    Y, X = torch.meshgrid(torch.linspace(0, h - 1, h), torch.linspace(0, w - 1, w))
    pix_grid = torch.zeros((X.shape[0] * X.shape[1], 2))
    pix_grid[:, 0] = X.flatten()
    pix_grid[:, 1] = Y.flatten()

    center = torch.tensor([w // 2., h // 2.])

    return pix_grid, center

def sample_points_on_plane(normal, depth, p2):
    """
    Sample points on plane:
    normal = 3 x h x w tensor
    depth  = 1 x h x w tensor
    p2     = 4 x 4 tensor
    returns
    sampled = h x w x 4 x 3 tensor
    points_3d = h x w x 3 tensor
    """
    MAGNITUDE = 1.0
    _, h, w = normal.shape
    if depth.ndim == 2:
            depth  = depth.unsqueeze(0)

    # Depth-Dependent point sampling on the plane
    # If we do not do this, sampling points on distant planes make the 4 points approximately the same resulting in incorrect homography
    mask       = depth[0] >  0                             # h x w
    MAG_SCALED = MAGNITUDE * torch.ones((h, w)).to(normal.device)
    MAG_SCALED[mask] += 0.1 * depth[0][mask]               # h x w
    MAGNITUDE  = MAG_SCALED.reshape(h*w, 1).repeat(1, 3)   # hw x 3

    # Camera matrix inverse
    p2_inv     = torch.linalg.inv(p2)

    # Construct backprojected points
    Y, X       = torch.meshgrid(torch.linspace(0, h-1, h), torch.linspace(0, w-1, w))
    X          = X.to(normal.device)
    Y          = Y.to(normal.device)

    points     = torch.concat((X.flatten().unsqueeze(0), Y.flatten().unsqueeze(0), depth.flatten().unsqueeze(0)), dim= 0)       # 3 x hw
    points_3d  = backproject_2d_pixels_in_4D_format(p2_inv, points= points, pad_ones= True)                                     # 4 x hw
    points_3d  = points_3d[:3].transpose(0, 1)                                                                                  # hw x 3
    # 4th Coefficient of Plane mX + nY + oZ + K = 0
    # Theorem 1 of DEVIANT, ECCV 2022.
    K          = -torch.sum(points_3d * normal.permute(1, 2, 0).reshape(-1, 3), dim= 1)                                         # hw

    # Plane Basis Vectors
    # Take probes: vectors which are not parallel to the normal vector.
    # eg: Normal along z-axis, probes = x and y unit vectors
    # Then take cross product of probes and normal to get the plane basis.
    normal     = normal.permute(1, 2, 0).reshape(-1, 3)   # hw x 3
    max_arg    = torch.max(torch.abs(normal), dim= 1)[1]
    probes     = torch.zeros_like(normal).unsqueeze(1).repeat(1,2,1) # hw x 2 x 3
    probes [max_arg == 0] = torch.tensor([[0, 1, 0], [0, 0, -1]], dtype= normal.dtype, device= normal.device)
    probes [max_arg == 1] = torch.tensor([[-1, 0, 0], [0, 0, 1]], dtype= normal.dtype, device= normal.device)
    probes [max_arg == 2] = torch.tensor([[1, 0, 0], [0, -1, 0]], dtype= normal.dtype, device= normal.device)
    basis      = torch.cross(normal.unsqueeze(1).repeat(1, 2, 1).reshape(-1, 3), probes.reshape(-1, 3)).reshape(-1, 2, 3)  # hw x 2 x 3

    # Sample points in the plane using basis vectors.
    sampled =  torch.zeros_like(points_3d).unsqueeze(1).repeat(1, 4, 1)  # hw x 4 x 3
    sampled[:, 0] = points_3d  + MAGNITUDE * basis[:, 0] +  MAGNITUDE * basis[:, 1]
    sampled[:, 1] = points_3d  + MAGNITUDE * basis[:, 0] -  MAGNITUDE * basis[:, 1]
    sampled[:, 2] = points_3d  - MAGNITUDE * basis[:, 0] -  MAGNITUDE * basis[:, 1]
    sampled[:, 3] = points_3d  - MAGNITUDE * basis[:, 0] +  MAGNITUDE * basis[:, 1]

    # Reshape vector
    sampled = sampled.reshape(h, w, 4, 3)
    points_3d = points_3d.reshape(h, w, 3)
    points    = points.permute(1, 0).reshape(h, w, 3)
    normal   = normal.reshape(h, w, 3)
    depth    = depth[0].reshape(h, w)
    K        = K.reshape(h, w)

    return sampled, points_3d

def sample_points_on_plane_batched(normal, depth, p2):
    """
    Sample points on plane:
    normal    = b x 3 x h x w tensor
    depth     = b x h x w tensor
    p2        = 4 x 4 tensor
    returns
    sampled   = b x h x w x 4 x 3 tensor
    points_3d = b x h x w x 3 tensor
    """
    MAGNITUDE = 1.0
    b, _, h, w = normal.shape
    if depth.ndim == 2:
        depth  = depth.unsqueeze(0)

    # Depth-Dependent point sampling on the plane
    # If we do not do this, sampling points on distant planes make the 4 points approximately the same resulting in incorrect homography
    mask       = depth >  0                           # b x h x w
    MAG_SCALED = MAGNITUDE * torch.ones((b, h, w)).to(normal.device)
    MAG_SCALED[mask] += 0.1 * depth[mask]            # b x h x w
    MAGNITUDE  = MAG_SCALED.reshape(b*h*w, 1).repeat(1, 3) # bhw x 3

    # Camera matrix inverse
    p2_inv     = torch.linalg.inv(p2)

    # Construct backprojected points
    Y, X       = torch.meshgrid(torch.linspace(0, h-1, h), torch.linspace(0, w-1, w))
    X          = X.to(normal.device).unsqueeze(0).repeat(b, 1, 1)
    Y          = Y.to(normal.device).unsqueeze(0).repeat(b, 1, 1)

    points     = torch.concat((X.flatten().unsqueeze(0), Y.flatten().unsqueeze(0), depth.flatten().unsqueeze(0)), dim= 0)       # 3 x bhw
    points_3d  = backproject_2d_pixels_in_4D_format(p2_inv, points= points, pad_ones= True)                                     # 4 x bhw
    points_3d  = points_3d[:3].transpose(0, 1)                                                                                  # bhw x 3
    # 4th Coefficient of Plane mX + nY + oZ + K = 0
    # Theorem 1 of DEVIANT, ECCV 2022.
    K          = -torch.sum(points_3d * normal.permute(0, 2, 3, 1).reshape(-1, 3), dim= 1)                                         # bhw

    # Plane Basis Vectors
    # Take probes: vectors which are not parallel to the normal vector.
    # eg: Normal along z-axis, probes = x and y unit vectors
    # Then take cross product of probes and normal to get the plane basis.
    normal     = normal.permute(0, 2, 3, 1).reshape(-1, 3)   # bhw x 3
    max_arg    = torch.max(torch.abs(normal), dim= 1)[1]
    probes     = torch.zeros_like(normal).unsqueeze(1).repeat(1,2,1) # bhw x 2 x 3
    probes [max_arg == 0] = torch.tensor([[0, 1, 0], [0, 0, -1]], dtype= normal.dtype, device= normal.device)
    probes [max_arg == 1] = torch.tensor([[-1, 0, 0], [0, 0, 1]], dtype= normal.dtype, device= normal.device)
    probes [max_arg == 2] = torch.tensor([[1, 0, 0], [0, -1, 0]], dtype= normal.dtype, device= normal.device)
    basis      = torch.cross(normal.unsqueeze(1).repeat(1, 2, 1).reshape(-1, 3), probes.reshape(-1, 3)).reshape(-1, 2, 3)  # bhw x 2 x 3

    # Sample points in the plane using basis vectors.
    sampled =  torch.zeros_like(points_3d).unsqueeze(1).repeat(1, 4, 1)  # bhw x 4 x 3
    sampled[:, 0] = points_3d  + MAGNITUDE * basis[:, 0] +  MAGNITUDE * basis[:, 1]
    sampled[:, 1] = points_3d  + MAGNITUDE * basis[:, 0] -  MAGNITUDE * basis[:, 1]
    sampled[:, 2] = points_3d  - MAGNITUDE * basis[:, 0] -  MAGNITUDE * basis[:, 1]
    sampled[:, 3] = points_3d  - MAGNITUDE * basis[:, 0] +  MAGNITUDE * basis[:, 1]

    # Reshape vector
    sampled = sampled.reshape(b, h, w, 4, 3)
    points_3d = points_3d.reshape(b, h, w, 3)
    points    = points.permute(1, 0).reshape(b, h, w, 3)
    normal   = normal.reshape(b, h, w, 3)
    depth    = depth.reshape(b, h, w)
    K        = K.reshape(b, h, w)

    return sampled, points_3d