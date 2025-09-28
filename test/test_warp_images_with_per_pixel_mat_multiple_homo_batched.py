"""
  Sample Run:
  python test/.py
"""
import copy
import os, sys
sys.path.append(os.getcwd())

import os.path as osp
import glob
import numpy as np
import torch
import torch.nn as nn

np.set_printoptions   (precision= 2, suppress= True)
torch.set_printoptions(precision= 2, sci_mode= False)

from plot.common_operations import *
import plot.plotting_params as params
import matplotlib

from lib.helpers.util import init_torch, read_RGB, read_depth, read_normal, inch_2_meter, ee_error
from lib.helpers.homography_helper import get_intrinsics_from_fov, sample_points_on_plane, get_per_pixel_image_homo_batched, _warp_images_with_per_pixel_mat, invert_Homography

import random


def foo(folder, ty_inch= -30, indices= None):
    tx = 0
    tz = 0.
    ty = inch_2_meter(ty_inch)
    print(ty)

    print("Reading {}".format(folder))
    image_folder  = folder + "image"
    depth_folder  = folder + "depth"
    normal_folder = folder + "normal"
    seman_folder  = folder + "seman"

    image_files = sorted(glob.glob(image_folder + "/*.jpg"))
    if indices is not None:
        print("Selecting {} images".format(len(indices)))
        image_files = [image_files[i] for i in indices]

    depth_files = sorted(glob.glob(depth_folder + "/*.npy"))
    if indices is not None:
        depth_files = [depth_files[i] for i in indices]

    normal_files = sorted(glob.glob(normal_folder + "/*.png"))
    if indices is not None:
        normal_files = [normal_files[i] for i in indices]

    seman_files = sorted(glob.glob(seman_folder + "/*.npy"))
    if indices is not None:
        seman_files = [seman_files[i] for i in indices]

    ee_pts = np.zeros((len(image_files), ))
    ee_inf = np.zeros((len(image_files), ))
    ee_sam = np.zeros((len(image_files), ))

    for i, (imf, df, nf, sf) in enumerate(zip(image_files, depth_files, normal_files, seman_files)):

        image  = read_RGB(imf, device, downsample, smooth)
        depth  = read_depth(df, device, downsample, smooth)  # 1 x h x w
        normal = read_normal(nf, device, downsample, smooth) # 3 x h x w
        seman  = read_depth(sf, device, downsample, smooth)  # 3 x h x w

        trn_image_path = imf.replace("height0", "height30")
        trn_depth_path =  df.replace("height0", "height30")
        trn_normal_path = nf.replace("height0", "height30")
        depth1 = read_depth(trn_depth_path, device, downsample, smooth)  # 1 x h x w
        normal1 = read_normal(trn_normal_path, device, downsample, smooth) # 3 x h x w

        # print(imf, df, nf)

        intrinsics = get_intrinsics_from_fov(w= w, h= h, fov_degree= fov_degree)
        p2         = torch.eye(4).float()
        p2[:3, :3] = torch.from_numpy(intrinsics).float()
        p2         = p2.to(normal.device)

        # Batching
        depth  = depth.repeat(2, 1, 1)
        normal = normal.unsqueeze(0).repeat(2, 1, 1, 1)  # 2 x 3 x h x w

        with torch.no_grad():
            if invert_Homo:
                Hom01_pts = get_per_pixel_image_homo_batched(depth, normal, tx = 0., ty= -ty, tz = 0., fov_degree= fov_degree, method= "points")[0]
                Hom10_pts = invert_Homography(Hom01_pts, depth)
                Hom01_inf = get_per_pixel_image_homo_batched(depth, normal, tx = 0., ty= -ty, tz = 0., fov_degree= fov_degree, method= "infinite_plane")[0]
                Hom10_inf = invert_Homography(Hom01_inf, depth)
                Hom01_sam = get_per_pixel_image_homo_batched(depth, normal, tx = 0., ty= -ty, tz = 0., fov_degree= fov_degree, method= "plane_sampling")[0]
                Hom10_sam = invert_Homography(Hom01_sam, depth)
            else:
                Hom10_pts = get_per_pixel_image_homo_batched(depth, normal, tx = 0., ty= ty, tz = 0., fov_degree= fov_degree, method= "points")[0]
                Hom10_inf = get_per_pixel_image_homo_batched(depth, normal, tx = 0., ty= ty, tz = 0., fov_degree= fov_degree, method= "infinite_plane")[0]
                Hom10_sam = get_per_pixel_image_homo_batched(depth, normal, tx = 0., ty= ty, tz = 0., fov_degree= fov_degree, method= "plane_sampling")[0]

            img0        = read_RGB(imf, downsample= downsample)
            img1        = read_RGB(trn_image_path, downsample= downsample)

            # We are running with batch of two images. Take one of the batches
            Hom10_pts   = Hom10_pts[1].unsqueeze(0)
            Hom10_inf   = Hom10_inf[1].unsqueeze(0)
            Hom10_sam   = Hom10_sam[1].unsqueeze(0)

            fixed = torch.eye(3).to(device)
            fixed[1,2]  = -20
            # fixed[1, 1] = 2.0
            # fixed[0, 0] = 2.0
            # Hom10_inf  = fixed.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1, h, w, 1, 1)

            img0_tensor = torch.Tensor(img0).permute(2, 0, 1).to(device).unsqueeze(0)                     # b x 3 x h x w
            img1_tensor = torch.Tensor(img1).permute(2, 0, 1).to(device).unsqueeze(0)                     # b x 3 x h x w
            img0_warp_pts_t = _warp_images_with_per_pixel_mat(img0_tensor, Hom10_pts)        # b x 3 x h x w
            img0_warp_pts   = img0_warp_pts_t.cpu().float()[0].permute(1, 2, 0).numpy().astype(np.uint8)  # h x w x 3

            img0_warp_inf_t = _warp_images_with_per_pixel_mat(img0_tensor, Hom10_inf)        # b x 3 x h x w
            img0_warp_inf   = img0_warp_inf_t.cpu().float()[0].permute(1, 2, 0).numpy().astype(np.uint8)  # h x w x 3

            img0_warp_sam_t = _warp_images_with_per_pixel_mat(img0_tensor, Hom10_sam)        # b x 3 x h x w
            img0_warp_sam   = img0_warp_sam_t.cpu().float()[0].permute(1, 2, 0).numpy().astype(np.uint8)  # h x w x 3

        normal_np = 0.5*(normal[0].cpu().permute(1,2,0).numpy()+1)*255
        normal_np = normal_np.astype(np.uint8)

        if show_points:
            plt.figure(figsize= (20,4), dpi= 200)
            plt.subplot(r,c,1)
            plt.imshow(img0)
            plt.title('Original')
            plt.axis('off')

            plt.subplot(r,c,2)
            plt.imshow(normal_np)
            plt.title('Normal')
            plt.axis('off')

            plt.subplot(r,c,3)
            plt.imshow(img0_warp_inf)
            plt.title('Infinite')
            plt.axis('off')

            plt.subplot(r,c,4)
            plt.imshow(img0_warp_pts)
            plt.title('Points')
            plt.axis('off')

            plt.subplot(r,c,5)
            plt.imshow(img0_warp_sam)
            plt.title('Finite')
            plt.axis('off')

            plt.subplot(r,c,6)
            plt.imshow(img1)
            plt.title('GT')
            plt.axis('off')

            savefig(plt, "images/warp_images_with_per_pixel_homo_multiple_homo_batched.png")
            plt.show()
            plt.close()


        mask = torch.zeros((1, 3, h, w), dtype= torch.bool).to(device)
        mask[:, : , h//4:3*h//4, w//4:3*w//4] = True
        # mask = None
        ee_pts[i] = ee_error(img0=img0_warp_pts_t, img1=img1_tensor, mask= mask).item()
        ee_inf[i] = ee_error(img0=img0_warp_inf_t, img1=img1_tensor, mask= mask).item()
        ee_sam[i] = ee_error(img0=img0_warp_sam_t, img1=img1_tensor, mask= mask).item()
        # print("hello")


    print("Error in Inf= {:.2f}".format(np.mean(ee_inf)))
    print("Error in Pts= {:.2f}".format(np.mean(ee_pts)))
    print("Error in Sam= {:.2f}".format(np.mean(ee_sam)))



# ==================================================================================================
# Starts here
# ==================================================================================================
init_torch(0, 0)
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
NUM_FILES     = 50
# val_0_folder  = "data/carla/height0/training/"
val_0_folder  = "data/carla/height0/validation/"
downsample    = 4
invert_Homo   = True
smooth        = False

fov_degree= 64.56
w = 512 // downsample
h = 512 // downsample
r = 1
c = 6

first_folder = val_0_folder + "depth"
num_files    = len(sorted(glob.glob(first_folder + "/*.npy")))

if "training" in val_0_folder:
    indices     = [40]
    show_points = True
    height_inches_list = [-30]
else:
    indices     = sorted(random.sample(range(num_files), NUM_FILES))
    show_points = False
    height_inches_list = [-30]#np.arange(6, 36, 6)

foo(val_0_folder, height_inches_list[0], indices)
