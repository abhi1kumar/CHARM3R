import numpy as np
import torch
import torch.nn as nn
import time
import logging

from torchvision.ops import deform_conv2d
from lib.helpers.homography_helper import hom_to_grid_offset, grid_offset_to_deform_offset, _warp_images_with_per_pixel_mat, invert_Homography, get_per_pixel_image_homo_batched
from lib.helpers.util import inch_2_meter


class DepthReciprocalConv2d(nn.Module):
    def __init__(self, in_channels= 1, out_channels= 1, kernel_size= 3, padding= 1, stride= 1, dilation= 1, bias= False, pass_warped_images= True, Dmin= 1, Dmax= 50, focal= 50., style= "deformable", formulation= "depth", geometry_method = "max", project= True, project_operator="min", full_homo= False, original_image_size= [512, 512], conv= None):
        super().__init__()
        if conv is not None:
            # Use properties of conv to initialize
            in_channels  = conv.in_channels
            out_channels = conv.out_channels
            kernel_size  = conv.kernel_size[0] if type(conv.kernel_size) == tuple else conv.kernel_size
            padding      = conv.padding[0]     if type(conv.padding)     == tuple else conv.padding
            stride       = conv.stride[0]      if type(conv.stride)      == tuple else conv.stride
            dilation     = conv.dilation[0]    if type(conv.dilation)    == tuple else conv.dilation
            bias         = conv.bias is not None

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding
        self.dilation     = dilation
        self.bias         = bias
        self.pass_warped_images = pass_warped_images
        self.style = style
        self.geometry_method = geometry_method
        if self.geometry_method == "convolve":
            self.geometry_conv = nn.Conv2d(self.in_channels, 1, kernel_size= (kernel_size, kernel_size), padding= self.padding, stride=(1,1), dilation= (self.dilation, self.dilation), bias= True)

        # Depth Reciprocal Settings
        self.Dmin = Dmin
        self.Dmax = Dmax
        self.focal = focal
        self.formulation = formulation
        self.original_image_size = original_image_size # w x h

        output  = "Lower_branch_style = {} Geometry_Method = {} Formulation = {} ".format(self.style, self.geometry_method, self.formulation)
        output += "Warping = {} ".format(self.pass_warped_images)

        # Upper Branch Convolution
        self.conv  = nn.Conv2d(self.in_channels, self.out_channels, kernel_size= (kernel_size, kernel_size), padding= self.padding, stride=(self.stride,self.stride), dilation= (self.dilation, self.dilation), bias= bias)
        # Do initialization
        if conv is not None:
            self.init_with_conv(conv)

        if self.style == "deformable":
            kernel_grid_Y, kernel_grid_X  = torch.meshgrid(torch.linspace(0, self.kernel_size-1, self.kernel_size), torch.linspace(0, self.kernel_size-1, self.kernel_size))
            self.kernel_grid = torch.concat((kernel_grid_X.reshape(self.kernel_size*self.kernel_size, 1), kernel_grid_Y.reshape(self.kernel_size*self.kernel_size, 1)), dim= 1)  # N x 2

        # Normal Settings
        self.full_homo = full_homo
        if self.full_homo:
            output += "Homography= Full "
            self.normal_conv = nn.Conv2d(self.in_channels, 3, kernel_size= (kernel_size, kernel_size), padding= self.padding, stride=(1,1), dilation= (self.dilation, self.dilation), bias= True)
            self.init_normal()
        else:
            output += "Homography= Frontal "

        # Project Settings
        self.project = project
        self.project_operator = project_operator
        if self.project:
            output += "Projecting with {}".format(self.project_operator)

        if logging._startTime < time.time():
            logging.info(output)
        else:
            print(output)

    def init_normal(self):
        with torch.no_grad():
            # Assign normals as z-plane
            self.normal_conv.bias.copy_(nn.Parameter(torch.tensor([0, 0, 1.])))

    def normalize_normal(self, norm_in):
        # normalize norm to be in the range [-1, 1]
        # See https://github.com/baegwangbin/surface_normal_uncertainty/blob/main/models/submodules/submodules.py#L64-L70
        norm_x, norm_y, norm_z = torch.split(norm_in, 1, dim=1)
        magnitude = torch.sqrt(norm_x ** 2.0 + norm_y ** 2.0 + norm_z ** 2.0) + 1e-3
        norm_out  = torch.cat([norm_x / magnitude, norm_y / magnitude, norm_z / magnitude], dim=1)

        return norm_out

    def init_with_conv(self, conv):
        with torch.no_grad():
            self.conv.weight.copy_(conv.weight)
            if self.bias:
                self.conv.bias.copy_(conv.bias)

    def geometry(self, x):
        """
        Return geometry (depth/disparity) from input
        Input:  x torch.Tensor of shape B x C x H x W
        Output:   torch.Tensor of shape B x H x W
        """
        if self.geometry_method == "max":
            return torch.max(x, dim= 1)[0]
        elif self.geometry_method == "min":
            return torch.min(x, dim= 1)[0]
        elif self.geometry_method == "first":
            return x[:, 0]
        elif self.geometry_method == "last":
            return x[:, -1]
        elif self.geometry_method == "convolve":
            return self.geometry_conv(x)[:, 0]
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor):
        if isinstance(x, list):
            if len(x) == 3:
                x, extra, normal = x
            elif len(x) == 2:
                x, extra = x
                normal = None
        else:
            extra  = None
            normal = None

        b, c, h, w = x.shape
        if h != self.original_image_size[1]:
            downsample = self.original_image_size[1] / h

        x_upper = self.conv(x)

        # Calculate depth
        if extra is None:
            geometry = self.geometry(x)
        else:
            geometry = extra
        if self.formulation == "depth":
            depth     = torch.clamp(geometry, self.Dmin, self.Dmax)      # B x h x w
        elif self.formulation == "disparity":
            disparity = torch.clamp(geometry, 1/self.Dmax, 1/self.Dmin)  # B x h x w
            depth     = 1.0/ disparity

        # Calculate Homography
        if self.full_homo:
            if normal is None:
                normal = self.normal_conv(x)
            normal = self.normalize_normal(normal)
            Hom01  = get_per_pixel_image_homo_batched(depth, normal, tx = 0., ty= -inch_2_meter(-30), tz = 0., fov_degree= 64.56, method= "plane_sampling")[0]  # B x h x w x 3 x 3

        else:
            Hom01   = torch.zeros((b, h, w, 3, 3), device= x.device, dtype= x.dtype)  # B x h x w x 3 x 3
            Hom01[:, :, :, 0, 0] = 1.
            Hom01[:, :, :, 1, 1] = 1.
            Hom01[:, :, :, 2, 2] = 1.
            Hom01[:, :, :, 1, 2] = self.focal/(downsample * depth)

        Hom10 = invert_Homography(Hom01, depth)

        # Warp Image
        if self.pass_warped_images:
            x2 = _warp_images_with_per_pixel_mat(x, mat=Hom10, transform_about_center=False)
        else:
            x2 = x

        if self.style == "vanilla":
            x_lower = self.conv(x2)
        elif self.style == "deformable":
            kernel_grid   = self.kernel_grid.type(x.dtype).to(x.device)  # N x 2
            kernel_center = torch.mean(kernel_grid, dim= 0)              # 2
            deform_offset = grid_offset_to_deform_offset(hom_to_grid_offset(Hom10, kernel_grid, kernel_center)) # B x 2N x h x w
            deform_offset = deform_offset[:, :, ::self.stride, ::self.stride]

            x_lower = deform_conv2d(input=x2,
                              offset=deform_offset,
                              weight=self.conv.weight,
                              bias=self.conv.bias,
                              padding=(self.padding, self.padding),
                              mask=None,
                              stride=(self.stride,self.stride),
                              dilation=(self.dilation, self.dilation))

        x_concat = torch.concat((x_upper.unsqueeze(1), x_lower.unsqueeze(1)), dim= 1) # B x 2 x h x w x c

        if self.project:
            x_concat = act(x_concat, self.project_operator)

        return x_concat

class DeformMax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return act(x)

def act(x, operator= "min"):
    if type(x) == torch.Tensor:
        if operator == "max":
            return torch.max(x, dim=1)[0]
        elif operator == "min":
            return torch.min(x, dim=1)[0]

    elif type(x) == np.ndarray:
        if operator == "max":
            return np.max(x, axis= 0)
        elif operator == "min":
            return np.min(x, axis= 0)

    else:
        raise NotImplementedError
