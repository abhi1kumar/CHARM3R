"""
    Warping of target domain images to source domain based on UniDrive, Li et al, ICLR 2025 (Under submission)
    https://github.com/ywyeli/UniDrive/blob/6313e5c98ce13c224b0cde0dbe52a0d97e4e657d/project/mmdet3d/datasets/pipelines/loading.py#L30-L161
"""
import os, sys
sys.path.append(os.getcwd())

import os.path as osp
import glob
import numpy as np
import torch
import torch.nn as nn

from numba import cuda, float32
import logging
cuda_logger = logging.getLogger('numba.cuda.cudadrv.driver')
cuda_logger.setLevel(logging.ERROR)  # only show error

@cuda.jit()
def project_pixels_cuda(output_image, output_weights, original_image,
                        original_intrinsics, original_extrinsics, original_extrinsics_inv,
                        new_intrinsics, new_extrinsics, new_extrinsics_inv, depth= 50):
    h, w = output_image.shape[0], output_image.shape[1]
    u_new, v_new = cuda.grid(2)

    Bottom_half = 500

    if u_new < w and v_new < h:
        camera_height = new_extrinsics[2, 3] + 0.1
        if v_new > Bottom_half:  # Bottom half of the image
            # Assume all points are on the ground plane (Z_world = 0)
            X_new_camera = (u_new - new_intrinsics[0, 2]) / new_intrinsics[0, 0]
            Y_new_camera = (v_new - new_intrinsics[1, 2]) / new_intrinsics[1, 1]
            Z_new_camera = camera_height / Y_new_camera

            point_new_camera = cuda.local.array((4,), dtype=float32)
            point_new_camera[0] = X_new_camera * Z_new_camera
            point_new_camera[1] = camera_height  # Y is adjusted to maintain the ground plane assumption
            point_new_camera[2] = Z_new_camera
            point_new_camera[3] = 1.0
        else:
            # Convert pixel coordinates to camera space using the intrinsic matrix
            X_new_camera = (u_new - new_intrinsics[0, 2]) * depth / new_intrinsics[0, 0]
            Y_new_camera = (v_new - new_intrinsics[1, 2]) * depth / new_intrinsics[1, 1]
            Z_new_camera = depth

            point_new_camera = cuda.local.array((4,), dtype=float32)
            point_new_camera[0] = X_new_camera
            point_new_camera[1] = Y_new_camera
            point_new_camera[2] = Z_new_camera
            point_new_camera[3] = 1.0

        # Transform point from new camera coordinates to world coordinates
        point_world = cuda.local.array((4,), dtype=float32)
        for i in range(4):
            point_world[i] = 0.0
            for j in range(4):
                point_world[i] += new_extrinsics[i, j] * point_new_camera[j]

        # Transform point from world coordinates to original camera coordinates
        point_original_camera = cuda.local.array((4,), dtype=float32)
        for i in range(4):
            point_original_camera[i] = 0.0
            for j in range(4):
                point_original_camera[i] += original_extrinsics_inv[i, j] * point_world[j]

        if point_original_camera[2] > 0:
            point_original_image = cuda.local.array((3,), dtype=float32)
            for i in range(3):
                point_original_image[i] = 0.0
                for j in range(3):
                    point_original_image[i] += original_intrinsics[i, j] * point_original_camera[j]

            u_original = int(point_original_image[0] / point_original_image[2])
            v_original = int(point_original_image[1] / point_original_image[2])

            if 0 <= u_original < original_image.shape[1] and 0 <= v_original < original_image.shape[0]:
                for c in range(3):  # Copy each color channel
                    output_image[v_new, u_new, c] = original_image[v_original, u_original, c]
                    # cuda.atomic.add(output_image, (v_new, u_new, c), original_image[v_original, u_original, c])
                output_weights[v_new, u_new] += 1


def run_project_pixels_cuda(original_image, original_intrinsics, original_extrinsics, new_extrinsics, new_intrinsics= None, Dis = 50):
    image_height = original_image.shape[0]
    image_width  = original_image.shape[1]
    output_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    output_weights = np.zeros((image_height, image_width), dtype=np.uint8)

    original_extrinsics_inv = np.linalg.inv(original_extrinsics)
    new_extrinsics_inv = np.linalg.inv(new_extrinsics)

    threadsperblock = (32, 32)
    blockspergrid_x = int(np.ceil(image_width / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(image_height / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    d_output_image = cuda.to_device(output_image)
    d_output_weights = cuda.to_device(output_weights)
    d_original_image = cuda.to_device(original_image)
    d_original_intrinsics = cuda.to_device(original_intrinsics)
    d_original_extrinsics = cuda.to_device(original_extrinsics)
    d_original_extrinsics_inv = cuda.to_device(original_extrinsics_inv)
    d_new_intrinsics = cuda.to_device(original_intrinsics) if new_intrinsics is None else cuda.to_device(new_intrinsics)
    d_new_extrinsics = cuda.to_device(new_extrinsics)
    d_new_extrinsics_inv = cuda.to_device(new_extrinsics_inv)

    project_pixels_cuda[blockspergrid, threadsperblock](
        d_output_image, d_output_weights,  d_original_image,
        d_original_intrinsics, d_original_extrinsics,
        d_original_extrinsics_inv, d_new_intrinsics,
        d_new_extrinsics, d_new_extrinsics_inv, Dis
    )

    d_output_image.copy_to_host(output_image)
    d_output_weights.copy_to_host(output_weights)

    return output_image