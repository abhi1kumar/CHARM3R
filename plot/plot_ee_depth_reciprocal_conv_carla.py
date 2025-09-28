"""
    Sample Run:
    python .py
"""
import os, sys
sys.path.append(os.getcwd())

import os.path as osp
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

np.set_printoptions   (precision= 2, suppress= True)
torch.set_printoptions(precision= 2, sci_mode= False)

from plot.common_operations import *
import plot.plotting_params as params
import matplotlib
import random

from lib.helpers.util import init_torch, inch_2_meter, ee_error, read_RGB, pass_over_model, read_depth, read_normal
from lib.depth_reciprocal_lib.depth_reciprocal import DepthReciprocalConv2d, DeformMax
from lib.helpers.homography_helper import get_per_pixel_image_homo_batched, _warp_images_with_per_pixel_mat, invert_Homography
from lib.helpers.file_io import save_numpy

def foo(folder, ty_inch= -30, indices= None, model0= None, model1= None):
    tx = 0
    tz = 0.
    ty = inch_2_meter(ty_inch)

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

    assert len(depth_files) == len(normal_files)
    num_files  = len(depth_files)

    # hom_all = torch.zeros((num_files, h, w, 3, 3), device= device)
    ee_array_0 = np.zeros((num_files,))
    ee_array_1 = np.zeros((num_files,))
    for i, (imf, df, nf, sf) in enumerate(zip(image_files, depth_files, normal_files, seman_files)):
        trn_image_path  = imf.replace("height0", "height30")
        trn_depth_path  =  df.replace("height0", "height30")
        trn_normal_path =  nf.replace("height0", "height30")
        trn_seman_path  =  sf.replace("height0", "height30")

        image  = read_RGB(imf, device, downsample)   # 3 x h x w
        depth  = read_depth(df, device, downsample)  # b x h x w
        normal = read_normal(nf, device, downsample).unsqueeze(0) # b x 3 x h x w
        seman  = read_depth(sf, device, downsample)               # b x h x w

        x0         = image.unsqueeze(0)                                            # b x 3 x h x w
        x1         = read_RGB(trn_image_path, device, downsample).unsqueeze(0)     # b x 3 x h x w
        depth1     = read_depth(trn_depth_path, device, downsample)                # b x h x w
        normal1    = read_normal(trn_normal_path, device, downsample).unsqueeze(0) # b x 3 x h x w
        seman1     = read_depth(trn_seman_path, device, downsample).unsqueeze(0)   # b x 1 x h x w
        mask       = torch.logical_and(seman1 >= 12, seman1 <= 19).repeat(1, out_channels, 1,  1)

        # if i < 4:
        #     continue
        # print(i)

        Hom01 = get_per_pixel_image_homo_batched(depth, normal, ty= -ty, fov_degree= fov_degree, method= "plane_sampling")[0] # b x h x w x 3 x 3
        Hom10 = invert_Homography(Hom01, depth)  # b x h x w x 3 x 3

        with torch.no_grad():
            # Transform and Pass image
            M0_x0   = model0(x0)            # 1 x c x h x w
            T_M0_x0 = _warp_images_with_per_pixel_mat(M0_x0, mat=Hom10)   # 1 x 1 x h x w
            M0_x1   = model0(x1)            # 1 x c x h x w

            if formulation == "disparity":
                extra  = 1.0/depth
                extra1 = 1.0/depth1
            else:
                extra  = depth
                extra1 = depth1

            M1_x0   = model1([x0, extra, normal])   # 1 x c x h x w
            T_M1_x0 = _warp_images_with_per_pixel_mat(M1_x0, mat=Hom10)   # 1 x 1 x h x w
            M1_x1   = model1([x1, extra1, normal1])  # 1 x c x h x w

        if mask_forgd_obj:
            T_M0_x0 *= mask
            M0_x1   *= mask
            T_M1_x0 *= mask
            M1_x1   *= mask

        if mask_forgd_obj:
            ee_array_0[i] = ee_error(img0= T_M0_x0, img1= M0_x1, mask= mask).item()
            ee_array_1[i] = ee_error(img0= T_M1_x0, img1= M1_x1, mask= mask).item()
        else:
            ee_array_0[i] = ee_error(img0= T_M0_x0, img1= M0_x1).item()
            ee_array_1[i] = ee_error(img0= T_M1_x0, img1= M1_x1).item()

        if show_features:
            # Convert to numpy
            imOr = read_RGB(imf, downsample= downsample)
            imTr = read_RGB(trn_image_path, downsample= downsample)
            Or_mod0   = M0_x0[0, 0].cpu().detach().numpy()
            Or_mod0_t = T_M0_x0[0, 0].cpu().detach().numpy()
            Tr_mod0   = M0_x1[0, 0].cpu().detach().numpy()

            Or_mod1   = M1_x0[0, 0].cpu().detach().numpy()
            Or_mod1_t = T_M1_x0[0, 0].cpu().detach().numpy()
            Tr_mod1   = M1_x1[0, 0].cpu().detach().numpy()

            plt.figure(figsize=(c * 6, r * 6), dpi=params.DPI)
            plt.subplot(r, c, 1)
            plt.imshow(imOr)
            plt.axis('off')
            plt.title('Original')

            plt.subplot(r, c, 2)
            plt.imshow(Or_mod0, vmin= vmin, vmax= vmax, cmap= cmap)
            plt.axis('off')
            plt.title('CNN')

            plt.subplot(r, c, 3)
            plt.imshow(Or_mod0_t, vmin= vmin, vmax= vmax, cmap= cmap)
            plt.axis('off')
            plt.title('Trans CNN')

            plt.subplot(r, c, 4)
            plt.imshow(Or_mod1, vmin= vmin, vmax= vmax, cmap= cmap)
            plt.axis('off')
            plt.title('Our')

            plt.subplot(r, c, 5)
            plt.imshow(Or_mod1_t, vmin= vmin, vmax= vmax, cmap= cmap)
            plt.axis('off')
            plt.title('Trans Our')

            plt.subplot(r, c, c+1)
            plt.imshow(imTr)
            plt.axis('off')
            plt.title('New Image')

            plt.subplot(r, c, c+3)
            plt.imshow(Tr_mod0, vmin= vmin, vmax= vmax, cmap= cmap)
            plt.axis('off')
            plt.title('CNN')

            plt.subplot(r, c, c+5)
            plt.imshow(Tr_mod1, vmin= vmin, vmax= vmax, cmap= cmap)
            plt.axis('off')
            plt.title('Our')

            plt.show()
            plt.close()


        if (i+1)% 100 == 0 or (i+1) == num_files:
            print("{} images done".format(i+1))

    print("EE Model0 CNN Med = {:.2f} Mean = {:.2f}".format(np.median(ee_array_0),  np.mean(ee_array_0)   ))
    print("EE Model1 Our Med = {:.2f} Mean = {:.2f}".format(np.median(ee_array_1),  np.mean(ee_array_1)   ))


class myModel(nn.Module):
    def __init__(self, num_conv_layers= 2):
        super(myModel, self).__init__()
        self.layers = []
        self.num_conv_layers = num_conv_layers
        for l in range(num_conv_layers):
            self.layers.append(DepthReciprocalConv2d(conv= model0[2*l], pass_warped_images= pass_warped_images, style= style, Dmin= Dmin, Dmax= Dmax, focal= focal // downsample, formulation= formulation, full_homo= full_homo, project= True, project_operator= project_operator))

    def forward(self, x):
        if isinstance(x, list):
            x, extra, normal = x
        else:
            extra  = None
            normal = None

        for i in range(self.num_conv_layers):
            x = self.layers[i]([x, extra, normal])
            x = F.relu(x)

        return x

# ==================================================================================================
# Main Starts here
# ==================================================================================================
init_torch(0, 0)
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

r = 2
c = 5
vmin = 0
vmax = 1
cmap = 'magma'

fov_degree   = 64.56
project_operator= "max"
full_homo       = True
ks           = 3
in_channels  = 3
out_channels = 64
Dmin         = 1
Dmax         = 50
focal         = 307.98  # f * delta H/ Dmin
pad          = ks//2
mask_forgd_obj = False

# Images stuff
NUM_FILES     = 100
val_0_folder  = "data/carla/height0/validation/"
# val_0_folder  = "data/carla/height0/training/"

first_folder = val_0_folder + "depth"
num_files    = len(sorted(glob.glob(first_folder + "/*.npy")))

if "training" in val_0_folder:
    indices     = [40]
    show_features = True
    height_inches_list = [-30]
else:
    indices     = sorted(random.sample(range(num_files), NUM_FILES))
    show_features = False
    height_inches_list = [-30]

for style in ["deformable"]: # "vanilla"
    for pass_warped_images in [True, False]:
        for formulation in ["disparity", "depth"]:
            for downsample in [4, 8, 16]:
                for num_conv_layers in [1, 2, 3, 4]:
                    # w = 512 // downsample
                    # h = 512 // downsample

                    print("\n=========================================================================================================================")
                    print("Style = {} Warped = {} Formulation = {} DS= {} #Layers= {} Full_Homo= {} Project_Op= {}".format(style, pass_warped_images, formulation, downsample, num_conv_layers, full_homo, project_operator))
                    print("=========================================================================================================================")
                    # Model initializations
                    module_list = []
                    for l in range(num_conv_layers):
                        if l == 0:
                            module_list.append(nn.Conv2d(in_channels, out_channels, (ks, ks), padding=(pad, pad), stride= [1, 1], bias= False))
                        else:
                            module_list.append(nn.Conv2d(out_channels, out_channels, (ks, ks), padding=(pad, pad), stride= [1, 1], bias= False))
                        module_list.append(nn.ReLU())
                    model0 = nn.Sequential(*module_list)

                    for l in range(num_conv_layers):
                        if l == 0:
                            conv_weight       = torch.zeros((out_channels, in_channels, ks, ks))
                        else:
                            conv_weight       = torch.zeros((out_channels, out_channels, ks, ks))
                        conv_weight[:, :] = 1.0/(ks*ks*in_channels)
                        model0[l*2].weight    = torch.nn.Parameter(conv_weight)

                    model1 = myModel(num_conv_layers= num_conv_layers)

                    # Send to cuda
                    model0 = model0.to(device)
                    model1 = model1.to(device)
                    for l in range(num_conv_layers):
                        model1.layers[l].conv.weight.data = model1.layers[l].conv.weight.data.cuda()


                    for ty_inch in height_inches_list:
                        foo(folder= val_0_folder, ty_inch= ty_inch, indices= indices, model0= model0, model1= model1)
