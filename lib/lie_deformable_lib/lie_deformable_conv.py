import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d
import logging

from lib.helpers.homography_helper import _warp_images_with_mat, projective_transform, get_pix_grid_center
torch.linalg.inv(torch.ones((1, 1), device="cuda:0"))

def lie_weights_to_hom(lie_weights, dom, generator= False):
    """
    Inputs:
        lie_weights = tensor B x h x w
        dom         = tensor 3 x 3
        generator   = bool whether the dominant transforms are the Lie algebra generators
    Output:
        Hom_all     = tensor B x h x w x 3 x 3
    """
    B, h, w  = lie_weights.shape
    if generator:
        # Linearly combine lie_weights in the tangent space and take matrix exponential to map to
        # manifold space
        lie_weights = lie_weights[:, :, :, None, None]
        dom         = dom[None, None, None, :, :]
        lie_vectors = torch.mul(lie_weights, dom)
        Hom_all     = torch.matrix_exp(lie_vectors)
    else:
        d,Pc     = torch.linalg.eig(dom)
        P        = torch.real(Pc)         # 3 x 3
        Pinv     = torch.linalg.inv(P)    # 3 x 3
        eig_val  = torch.real(d)          # 3

        lie_weights  = lie_weights.reshape(-1)
        out_diag_vec = torch.pow(eig_val[None, :], lie_weights[:, None]) # Bhw x 3
        out_diag     = torch.diag_embed(out_diag_vec)                    # Bhw x 3 x 3
        P_all        = P   .unsqueeze(0).repeat(B*h*w, 1, 1)             # Bhw x 3 x 3
        Pinv_all     = Pinv.unsqueeze(0).repeat(B*h*w, 1, 1)             # Bhw x 3 x 3
        Hom_all      = torch.bmm(torch.bmm(P_all, out_diag), Pinv_all)   # Bhw x 3 x 3
        Hom_all      = Hom_all.reshape(B, h, w, 3, 3)

    return Hom_all

def patch_warp_unpatch(input, Hom_all, patchify, unpatchify):
    # Apply homographies
    patch1     = patchify(input)
    patch_homo = _warp_images_with_mat(input_images=patch1, mat=Hom_all)
    output2    = unpatchify(patch_homo)

    return output2

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

def normalize_coord(U, V, w, h):
    U = U.float() / (w - 1)
    V = V.float() / (h - 1)
    U = U * 2 - 1
    V = V * 2 - 1

    return U, V

def invert_coord(U, V, w, h):
    EPS = 1e-2
    U = U.float() - w/2.
    V = V.float() - h/2.
    U = 1.0/(torch.abs(U) + EPS)
    V = 1.0/(torch.abs(V) + EPS)

    return U, V

def adapt_model(model, shift_coord_conv, eval_gd_homo= None):
    for child_name, child in model.named_children():
        if isinstance(child, LieDeformableConv2D):
            child.adapt_in_eval(shift_coord_conv, eval_gd_homo)
        else:
            adapt_model(child, shift_coord_conv, eval_gd_homo)

def reset_adapt_model(model):
    for child_name, child in model.named_children():
        if isinstance(child, LieDeformableConv2D):
            child.reset_adapt_in_eval()
        else:
            reset_adapt_model(child)

class LieDeformableConv2D(nn.Module):
    """
    Based on
    https://github.com/developer0hye/PyTorch-Deformable-Convolution-v2/blob/main/dcn.py
    """
    def __init__(self, in_channels= 1, out_channels= 1, kernel_size= 3, stride= 1, padding= 1, dilation= 1,
                 use_bias= False, conv= None, style= 'lie', version= 1, gd_homo= None, init_offset= 0.0,
                 clamp_min= 0.0, clamp_max= 1.0,
                 coord_conv= False, offset_append_coord= False, explicit_convolve_coord= False, use_normal= False,
                 coord_conv_style= "vanilla", coord_conv_merging= "append",
                 coord_fourier_scale= 10, coord_fourier_channels= 8
                 ):
        """
        clamp_min, clamp_max = Bound of Lie Predictor
        coord_conv = True adds a coordinate convolution calculation with each offset which can be changed based on extrinsics in inference
        offset_append_coord = Append coordinates to the Tangent space branch
        explicit_convolve_coord = Explicitly convolve coordinates and multiply with convolution outputs
        coord_conv_style = vanilla/inverted Use vanilla U, V or inverted U, V
        coord_conv_merging = append/sum/merge
        """

        super(LieDeformableConv2D, self).__init__()
        if conv is not None:
            # Use properties of conv to initialize
            in_channels  = conv.in_channels
            out_channels = conv.out_channels
            kernel_size  = conv.kernel_size[0] if type(conv.kernel_size) == tuple else conv.kernel_size
            stride       = conv.stride[0]      if type(conv.stride)      == tuple else conv.stride
            if 'Conv2dStaticSamePadding' in type(conv).__name__:
                padding  = conv.static_padding.padding[0]
            else:
                padding  = conv.padding[0]     if type(conv.padding)     == tuple else conv.padding
            dilation     = conv.dilation[0]    if type(conv.dilation)    == tuple else conv.dilation
            use_bias     = conv.bias is not None

        self.coord_conv              = coord_conv
        self.offset_append_coord     = offset_append_coord
        self.explicit_convolve_coord = explicit_convolve_coord
        self.coord_conv_style        = coord_conv_style
        self.coord_conv_merging      = coord_conv_merging

        # Main convolution branch
        self.coord_fourier_channels = coord_fourier_channels
        if self.coord_conv and self.coord_conv_merging == "append":
            if self.coord_conv_style == "npe":
                self.extra_in_channels = 8
            elif self.coord_conv_style == "fourier":
                self.extra_in_channels = self.coord_fourier_channels
            elif self.coord_conv_style == "oracle_depth" or self.coord_conv_style == "oracle_inv_depth":
                self.extra_in_channels = 1
            elif self.coord_conv_style == "oracle_normal" or self.coord_conv_style == "oracle_normal_inv_depth":
                self.extra_in_channels =  3
            elif "kprpe" in self.coord_conv_style:
                self.extra_in_channels = 3
                self.rpkpe       = nn.Linear(500, 3)
            elif self.coord_conv_style == "directions":
                self.extra_in_channels = 3
            elif self.coord_conv_style == "plucker":
                self.extra_in_channels = 3
            elif self.coord_conv_style == "directions_plucker":
                self.extra_in_channels = 6
            elif self.coord_conv_style == "oracle_canny":
                self.extra_in_channels = 1
            elif self.coord_conv_style == "oracle_normal_canny" or self.coord_conv_style == "oracle_normal_canny_inv":
                self.extra_in_channels = 3
            else:
                # raw pixel locations
                self.extra_in_channels = 2
        else:
            self.extra_in_channels = 0
        self.in_channels = in_channels + self.extra_in_channels
        self.out_channels= out_channels
        assert type(kernel_size) == tuple or type(kernel_size) == int
        kernel_size      = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride      = stride  if type(stride) == tuple else (stride, stride)
        self.padding     = padding
        self.dilation    = dilation
        self.use_bias    = use_bias

        # One of 'vanilla', 'deformable', 'lie', 'motion_basis'
        self.style       = style
        assert self.style in ["vanilla", "coord", "deformable", "lie", "motion_basis"]
        self.version     = version

        # Make the weights positive for lie/motion_basis so that only stretching occurs
        # "motion_basis" does not use clamp_max
        self.clamp_min   = clamp_min
        self.clamp_max   = clamp_max
        self.use_oracle  = use_normal
        self.init_offset = 0.0 if init_offset is None else init_offset
        if self.coord_conv:
            if self.coord_conv_style == "vanilla":
                logging.info("{}ing normalized coordinate conv...".format(self.coord_conv_merging))
            elif self.coord_conv_style == "inverted":
                logging.info("{}ing inverted coordinate conv...".format(self.coord_conv_merging))
            elif self.coord_conv_style == "npe":
                # Reference:
                # PLADE-Net: Towards Pixel-Level Accuracy for Self-Supervised Single-View Depth
                # Estimation with Neural Positional Encoding and Distilled Matting Loss
                # Bello et al, CVPR 2021
                # https://arxiv.org/pdf/2103.07362.pdf
                logging.info("{}ing neural positional encoding (NPE) as coordinate conv...".format(self.coord_conv_merging))
                conv1          = nn.Conv2d(in_channels = 2,
                                          out_channels= 8,
                                          kernel_size = 1,
                                          stride      = 1,
                                          padding     = 0,
                                          dilation    = 1,
                                          bias        = self.use_bias)
                conv2          = nn.Conv2d(in_channels = 8,
                                          out_channels= 8,
                                          kernel_size = 1,
                                          stride      = 1,
                                          padding     = 0,
                                          dilation    = 1,
                                          bias        = self.use_bias)
                self.coord_npe = nn.Sequential(conv1, nn.ELU(), conv2, nn.ELU())
            elif self.coord_conv_style == "fourier":
                # Reference:
                # Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains
                # Tancik et al., NeurIPS 2020
                # https://arxiv.org/pdf/2006.10739
                self.coord_fourier_scale = coord_fourier_scale
                logging.info("{}ing fourier features with scale= {:.2f} and channels= {:d} as coordinate conv..."
                             .format(self.coord_conv_merging, self.coord_fourier_scale, self.coord_fourier_channels))
                self.coord_fourier_basis = None
            elif "oracle" in self.coord_conv_style or "plucker" in self.coord_conv_style or "directions" in self.coord_conv_style:
                logging.info("{}ing {} features".format(self.coord_conv_merging, self.coord_conv_style))
            else:
                raise NotImplementedError
        if self.coord_conv and self.offset_append_coord:
            logging.info("Appending the tangent space branch with coordinates...")
        if self.use_oracle:
            logging.info("Using oracle lie coefficients / homography matrix...")

        self.conv = nn.Conv2d(in_channels = self.in_channels,
                              out_channels= self.out_channels,
                              kernel_size = kernel_size,
                              stride      = stride,
                              padding     = self.padding,
                              dilation    = self.dilation,
                              bias        = self.use_bias)
        if conv is not None:
            self.init_with_conv(conv)

        # Explicit convolving of coordinates
        if self.explicit_convolve_coord:
            logging.info("Adding extra convolution for explicitly convolving coordinates...")
            self.explicit_conv = nn.Conv2d(in_channels = 2,
                                           out_channels= 1,
                                           kernel_size = kernel_size,
                                           stride      = stride,
                                           padding     = self.padding,
                                           dilation    = self.dilation,
                                           bias        = True)
            nn.init.uniform_(self.explicit_conv.weight)
            nn.init.constant_(self.explicit_conv.bias, 0.)

        # Adapt during evaluation
        self.eval_adapt   = False
        self.shift_coord_conv    = None
        self.eval_gd_homo= None

        if self.style in ["deformable", "lie", "motion_basis"]:

            if self.coord_conv and self.offset_append_coord:
                offset_conv_in_channels = in_channels + 2
            else:
                offset_conv_in_channels = in_channels

            if self.style == "deformable":
                offset_conv_out_channels = 2 * kernel_size[0] * kernel_size[1]
                offset_use_bias = True
            elif self.style == "lie":
                offset_conv_out_channels = 1
                offset_use_bias = False
                assert gd_homo is not None
                self.gd_homo = gd_homo
            elif self.style == "motion_basis":
                offset_conv_out_channels = 8
                offset_use_bias = False
                # See https://user-images.githubusercontent.com/1344482/180954864-f9f0ac1b-f052-4a8f-bb3c-9f21f0ed46f3.png
                self.gd_homo = torch.eye(3)[None, :, :].repeat(8, 1, 1)
                self.gd_homo[1, 0, 1] = 1
                self.gd_homo[2, 0, 2] = 1
                self.gd_homo[3, 1, 0] = 1
                self.gd_homo[5, 1, 2] = 1
                self.gd_homo[6, 2, 0] = 1
                self.gd_homo[7, 2, 1] = 1
                self.gd_homo = self.gd_homo.reshape((8, 9)).transpose(1, 0) # 9 x 8

            if self.style in ["lie", "motion_basis"]:
                kh, kw         = self.kernel_size
                self.pix_grid, self.center = get_pix_grid_center(h= kh, w= kw)

            self.offset_conv = nn.Conv2d(offset_conv_in_channels,
                                         offset_conv_out_channels,
                                         kernel_size= kernel_size,
                                         stride     = stride,
                                         padding    = self.padding,
                                         dilation   = self.dilation,
                                         bias       = offset_use_bias)

            # Init offset_conv layer weight and bias
            if self.init_offset > 0.0:
                nn.init.uniform_(self.offset_conv.weight, 0, self.init_offset)
            else:
                nn.init.constant_(self.offset_conv.weight, 0.)
            if offset_use_bias:
                nn.init.constant_(self.offset_conv.bias, 0.)

            if self.version > 1.0:
                self.modulator_conv = nn.Conv2d(in_channels,
                                                1 * kernel_size[0] * kernel_size[1],
                                                kernel_size= kernel_size,
                                                stride     = stride,
                                                padding    = self.padding,
                                                dilation   = self.dilation,
                                                bias       = True)
                nn.init.constant_(self.modulator_conv.weight, 0.)
                nn.init.constant_(self.modulator_conv.bias, 0.)

    def init_with_conv(self, conv):
        with torch.no_grad():
            org_ch = self.in_channels - self.extra_in_channels
            logging.info("Initializing {} / {} channels".format(org_ch, self.in_channels))
            self.conv.weight[:, :org_ch].copy_(conv.weight)
            if self.use_bias:
                self.conv.bias.copy_(conv.bias)

    def adapt_in_eval(self, shift_coord_conv, eval_gd_homo= None):
        self.eval_adapt = True
        self.shift_coord_conv  = shift_coord_conv
        self.eval_gd_homo = eval_gd_homo

    def reset_adapt_in_eval(self):
        self.eval_adapt = False

    def forward(self, x, extra= None):
        if isinstance(x, tuple):
            x, extra = x
        b, c_in, h, w = x.shape
        device     = x.device
        lie_weights= None

        if self.coord_conv:
            if "oracle" in self.coord_conv_style or "plucker" in self.coord_conv_style or "directions" in self.coord_conv_style:
                assert extra is not None
                if extra.dim() == 4:
                    _, extra_c, extra_h, _ = extra.shape
                    downsample = extra_h // h
                    if downsample > 1:
                        extra = extra[:, :, ::downsample, ::downsample]
                elif extra.dim() == 3:
                    # Create Grid
                    V, U = torch.meshgrid(torch.linspace(0, h - 1, h), torch.linspace(0, w - 1, w))
                    V_norm = V.to(device)/h
                    U_norm = U.to(device)/w

                    V_norm = V_norm.reshape(1, 1, h, w) # b x 1 x h x w
                    U_norm = U_norm.reshape(1, 1, h, w) # b x 1 x h x w

                    V_norm = V_norm.repeat(b, 1, 1, 1)  # b x 1 x h x w
                    U_norm = U_norm.repeat(b, 1, 1, 1)  # b x 1 x h x w

                    # Create RPKPE
                    M = torch.cat([U_norm.reshape(b, h*w, 1), V_norm.reshape(b, h*w, 1)], dim= 2) # b x hw  x 2
                    P = extra                                                           # b x 250 x 2
                    # P has first coordinate as u and second as v
                    P[:, :, 0] /= w
                    P[:, :, 1] /= h
                    num_kpts = P.shape[1]
                    D = M.unsqueeze(2) - P.unsqueeze(1)                                # b x hw x 250 x 2
                    D = D.reshape(-1, 2*num_kpts)                                      # bhw x 500
                    extra = self.rpkpe(D)                                              # bhw x 3
                    extra = extra.reshape(b, h, w, 3).permute(0, 3, 1, 2)              # b x 3 x h x w

                if self.coord_conv_merging == "sum" or self.coord_conv_merging == "multiply":
                    if extra_c != c_in:
                        r_factor = c_in // extra_c
                        m_factor = c_in %  extra_c
                        extra = torch.cat([extra.repeat(1, r_factor, 1, 1), extra[:, :m_factor]], dim= 1)

                if self.coord_conv_merging == "append":
                    x_appended = torch.cat([x, extra], dim=1)  # b x (c_in + 1/3) x h x w
                elif self.coord_conv_merging == "sum":
                    x_appended = x + extra
                elif self.coord_conv_merging == "multiply":
                    x_appended = x * extra
                else:
                    raise NotImplementedError

            else:
                # Reference:
                # https://github.com/walsvid/CoordConv/blob/master/coordconv.py#L37-L65
                # Create Grid
                V, U = torch.meshgrid(torch.linspace(0, h-1, h), torch.linspace(0, w-1, w))
                V    = V.to(device)
                U    = U.to(device)

                # Adapt during evaluation
                if not self.training and self.eval_adapt and self.shift_coord_conv is not None:
                    self.shift_coord_conv = self.shift_coord_conv.to(device)
                    in_grid        = torch.cat((U.reshape(-1, 1), V.reshape(-1, 1)), dim= 1)     # N x 2
                    out_grid       = projective_transform(mat= self.shift_coord_conv, in_grid= in_grid) # N x 2
                    U = out_grid[:, 0].reshape((h, w))
                    V = out_grid[:, 1].reshape((h, w))
                V    = V[None, None, :, :]
                U    = U[None, None, :, :]

                if self.coord_conv_style == "vanilla" or self.coord_conv_style == "npe" or self.coord_conv_style == "fourier":
                    # Normalize
                    U_norm, V_norm = normalize_coord(U, V, w, h)
                elif self.coord_conv_style == "inverted":
                    U_norm, V_norm = invert_coord(U, V, w, h)
                else:
                    raise NotImplementedError
                V_norm         = V_norm.repeat(b, 1, 1, 1)                  # b x 1 x h x w
                U_norm         = U_norm.repeat(b, 1, 1, 1)                  # b x 1 x h x w

                if self.coord_conv_style == "npe":
                    # Pass through NPE
                    concat_coord = torch.cat([V_norm, U_norm], dim= 1)      # b x 2 x h x w
                    concat_coord = self.coord_npe(concat_coord)             # b x 8 x h x w
                    if self.coord_conv_merging == "append":
                        x_appended = torch.cat([x, concat_coord], dim= 1)   # b x (c_in + 8) x h x w

                elif self.coord_conv_style == "fourier":
                    # See https://github.com/tancik/fourier-feature-networks/blob/master/Demo.ipynb
                    # in particular input_mapping(x, B) function
                    if self.coord_fourier_basis is None:
                        self.coord_fourier_basis = torch.randn((h, 2), device= device) * self.coord_fourier_scale                        # h x 2

                    concat_coord = torch.cat([V_norm, U_norm], dim= 1)                                                                   # b x 2 x h x w
                    concat_coord = torch.matmul(2 * torch.pi * concat_coord.permute(0, 2, 3, 1), self.coord_fourier_basis.permute(1, 0)) # b x h x w x h
                    # Subsample
                    concat_coord = concat_coord[:, :, :, :self.coord_fourier_channels//2]                                                # b x h x w x f/2
                    concat_coord = concat_coord.permute(0, 3, 1, 2)                                                                      # b x f/2 x h x w
                    concat_coord = torch.cat([torch.sin(concat_coord), torch.cos(concat_coord)], dim= 1)                                 # b x f x h x w
                    if self.coord_conv_merging == "append":
                        x_appended = torch.cat([x, concat_coord], dim= 1)   # b x (c_in + f) x h x w
                else:
                    # take vanilla/inverted coordinates
                    if self.coord_conv_merging == "append":
                        # Append feature maps
                        x_appended = torch.cat([x, V_norm, U_norm], dim= 1)  # b x (c_in + 2) x h x w
                    else:
                        V_norm     = V_norm.repeat(1, c_in, 1, 1)
                        U_norm     = U_norm.repeat(1, c_in, 1, 1)
                        if self.coord_conv_merging == "sum":
                            x_appended = x + V_norm + U_norm                 # b x (c_in) x h x w
                        elif self.coord_conv_merging == "multiply":
                            x_appended = x * V_norm * U_norm

            if self.explicit_convolve_coord:
                explicit = self.explicit_conv(torch.cat((V_norm, U_norm), dim= 1))    # b x 1 x h x w
        else:
            x_appended = x

        # If there is stride, the feature map downsamples
        # p = updated h * updated w
        h //= self.stride[0]
        w //= self.stride[1]

        if self.style == "vanilla" or self.style == "coord":
            x = self.conv(x_appended)
        else:
            # Predict offset
            if self.coord_conv and self.offset_append_coord:
                offset_x = x_appended
            else:
                offset_x = x

            if self.style == "deformable":
                offset         = self.offset_conv(offset_x)
            elif self.style == "lie":
                if self.use_oracle:
                    # extra is the ground truth segmentation in this case
                    assert extra is not None
                    lie_weights= extra[:, None, :, :]                    # b x 1 x h x w
                    h_lie, w_lie = lie_weights.shape[2:]
                    if h_lie != h:
                        down_lie = h_lie // h
                        lie_weights = lie_weights[:, :, ::down_lie, ::down_lie]
                else:
                    lie_weights= self.offset_conv(offset_x)              # b x 1 x h x w
                    lie_weights= torch.clamp(lie_weights, min= self.clamp_min, max= self.clamp_max)       # b x 1 x h x w

                if not self.training and self.eval_adapt and self.eval_gd_homo is not None:
                    # Adapt during evaluation
                    Hom        = lie_weights_to_hom(lie_weights[:, 0], dom=self.eval_gd_homo.to(device))  # b x h x w x 3 x 3
                else:
                    Hom        = lie_weights_to_hom(lie_weights[:, 0], dom= self.gd_homo.to(device))  # b x h x w x 3 x 3

                self.pix_grid  = self.pix_grid.to(device)
                self.center    = self.center.to(device)
                grid_offset    = hom_to_grid_offset(Hom= Hom, kernel_grid= self.pix_grid, kernel_center= self.center) # b x h x w x n x 2
                offset         = grid_offset_to_deform_offset(grid_offset)                                  # b x 2*n x h x w
            elif self.style == "motion_basis":
                weights        = self.offset_conv(offset_x)                         # b x 8 x h x w
                weights        = torch.clamp(weights, min= self.clamp_min)          # b x 8 x h x w
                weights        = weights.permute(0, 2, 3, 1).reshape(-1, 8)         # b*h*w x 8
                weights_matrix = torch.diag_embed(weights)                          # b*h*w x 8 x 8
                ones_matrix    = torch.ones((b*h*w, 1, 8), dtype= x.dtype, device= x.device)
                weights_matrix = torch.cat((weights_matrix, ones_matrix), dim= 1)   # b*h*w x 9 x 8
                self.gd_homo   = self.gd_homo.to(device)                                # 9 x 8
                dom_temp       = self.gd_homo.clone()[None, :, :].repeat(b*h*w, 1, 1)   # b*h*w x 9 x 8
                Hom            = weights_matrix * dom_temp                          # b*h*w x 9 x 8
                Hom            = Hom.sum(dim= 2)                                    # b*h*w x 9
                Hom            = Hom.reshape(-1, 3, 3).reshape(b, h, w, 3, 3)       # b x h x w x 3 x 3

                self.pix_grid  = self.pix_grid.to(device)
                self.center    = self.center.to(device)
                grid_offset    = hom_to_grid_offset(Hom= Hom, kernel_grid= self.pix_grid, kernel_center= self.center) # b x h x w x n x 2
                offset         = grid_offset_to_deform_offset(grid_offset)                                  # b x 2*n x h x w

            # Predict mask
            if self.version > 1.0:
                # op = (n - (k * d - 1) + 2p / s)
                mask = 2. * torch.sigmoid(self.modulator_conv(x))
            else:
                mask = None

            x = deform_conv2d(input   = x_appended,
                              offset  = offset,
                              weight  = self.conv.weight,
                              bias    = self.conv.bias,
                              padding = (self.padding, self.padding),
                              mask    = mask,
                              stride  = self.stride,
                              dilation= (self.dilation, self.dilation))

        # Add linear regression term to outputs
        if self.coord_conv and self.explicit_convolve_coord:
            if self.style == "vanilla":
                 linear_regression = explicit                            # b x 1 x h x w
            elif self.style == "lie":
                linear_regression = explicit*lie_weights                 # b x 1 x h x w

            x = x + linear_regression.repeat(1, self.out_channels, 1, 1) # b x c x h x w

        # return x
        return [x, lie_weights]

    def extra_repr(self):
        s = 'style={style}, version={version}, coord_conv={coord_conv}'
        return s.format(**self.__dict__)
