"""
    Sample Run:
    python .py
"""
import os, sys

import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

import numpy as np
np.set_printoptions   (precision= 2, suppress= True)

from   plot.common_operations import *
import plot.plotting_params as params
import matplotlib
from scipy.stats import multivariate_normal
from lib.helpers.util import project_3d_points_in_4D_format

import torch
import torch.nn as nn

def flatten_grid(grid):
    Z, Y, X = grid
    XYZ = np.column_stack([X.flat, Y.flat, Z.flat])  # N x 3
    return XYZ

def project_covariance(covariance, jacobian, rotation, mode= "orthographic"):
    if mode == "orthographic":
        covar_2d = jacobian @ rotation @ covariance @ rotation.T @ jacobian.T

    return covar_2d[:2, :2]

def points(Xmin, Xmax, delta):
    return int((Xmax - Xmin)/delta) + 1

def create_grid(Xmax, Ymax, Zmax, Ymin= 0, Zmin= 0, delta= 0.25):
    Xmin = -Xmax
    x_   = np.linspace(Xmin, Xmax, points(Xmin, Xmax, delta))
    y_   = np.linspace(Ymin, Ymax, points(Ymin, Ymax, delta))
    z_   = np.linspace(Zmin, Zmax, points(Zmin, Zmax, delta))

    Z, Y, X = np.meshgrid(z_, y_, x_, indexing='ij')
    grid = [Z, Y, X]
    XYZ_flat = flatten_grid(grid)

    return grid, XYZ_flat

def conv_kernel(center, covariance, delta, size= 3):
    Xmax = size // 2
    Xmin = - Xmax

    Ymax = Xmax
    Ymin = Xmin

    Zmax = Xmax
    Zmin = Xmin

    grid, XYZ_flat  = create_grid(Xmax, Ymax, Zmax, Ymin, Zmin, delta= 1)
    output   = multivariate_normal.pdf(XYZ_flat, mean= center, cov=covariance) # N x 3
    kernel   = output.reshape(grid[0].shape)

    return  grid, XYZ_flat, kernel

def assign_input(grid, ground, depth, alpha = 1., beta = 1.):
    Z, Y, X = grid
    input  = np.zeros_like(Z)
    if beta > 0:
        input[Y == ground] = beta
    input[Z == depth]  = alpha

    return input

def ZYX_to_XZY(input):
    return input.transpose(2, 1, 0)

def norm(input, vmin, vmax):
    return (input - vmin)/ (vmax - vmin)

def extract_XYZ(input):
    Z = input[:, 0]
    Y = input[:, 1]
    X = input[:, 2]

    return X, Y, Z

def plot_3d_scatter(XYZ_flat, input_np, ax= None, vmin= 0):
    Z, Y, X = extract_XYZ(XYZ_flat)
    vmax = np.max(input_np.flatten())
    norm_input = norm((input_np).flatten(), vmin, vmax)
    colors = plt.cm.Purples(norm_input)
    size_data = 200 * np.ones_like(Z) * norm_input

    x = ax.scatter(X, Z, Y, c=colors, s=size_data, marker='s', vmin=vmin, vmax=vmax)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.invert_zaxis()

    return x


# ==================================================================================================
# Main starts here
# ==================================================================================================
ground = 1
depth  = 3
alpha  = 1.
beta   = 0.5
Xmax   = 1
Ymax   = ground + 1
Zmax   = depth + 1
delta  = 0.25
ksize  = 3
kpad   = ksize//2
f      = 100
h      = 256

Cin    = 1
Cout   = 1
B      = 1
vmin   = 0

r      = 1
c      = 3




jacobian   = np.array([[1, 0, 0], [0, 1., 0], [0, 0, 1]])
covariance = np.array([[1, 0, 0], [0, 3., 0], [0, 0, 1]])*0.5
rotation   = np.eye(3)
covar_2d   = project_covariance(covariance, jacobian, rotation)

# print(covar_2d)

grid, igrid_flat           = create_grid(Xmax, Ymax, Zmax, Zmin= 1, delta= delta)
kgrid, kgrid_flat, kernel = conv_kernel(center= (0, 0, 0), covariance= covariance, delta= delta, size= ksize)

input_np = assign_input(grid, ground, depth, alpha = alpha, beta = beta)
input = torch.from_numpy(input_np[np.newaxis, np.newaxis]).float().cuda()
model = nn.Sequential(torch.nn.Conv3d(in_channels= Cin, out_channels= Cout, kernel_size= ksize, stride= 1, padding= kpad)).cuda()
# weight = torch.ones((1, 1, ksize, ksize, ksize)).float().cuda()
print(kernel)
weight = torch.from_numpy(kernel[np.newaxis, np.newaxis]).float().cuda()
model[0].weight  = torch.nn.Parameter(weight)

output = model(input)
output_np = output.float().cpu().detach().numpy()[0, 0]
print(input.shape)
print(output.shape)


# Project output grid to 2D
p2 = np.array([[f, 0, h/2, 0], [0, f, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
proj_grid     = project_3d_points_in_4D_format(p2, points_4d= igrid_flat.T, pad_ones= True)[:2].T # N x 2
proj_grid     = proj_grid.astype(np.uint8)
input_np_flat = input_np.flatten()

image = np.zeros((2*h, 2*h))

# ==================================================================================================
# Plot here
# ==================================================================================================
fig = plt.figure(dpi= params.DPI, figsize= (18, 6))
ax  = fig.add_subplot(r,c,1, projection='3d')
x   = plot_3d_scatter(XYZ_flat= igrid_flat, input_np= input_np, ax= ax)
plt.title('Input')
# plt.colorbar(x)

ax  = fig.add_subplot(r,c,2, projection='3d')
x   = plot_3d_scatter(XYZ_flat= kgrid_flat, input_np= kernel, ax= ax)
plt.title('Kernel')
# plt.colorbar(x)

ax  = fig.add_subplot(r,c,3, projection='3d')
x   = plot_3d_scatter(XYZ_flat= igrid_flat, input_np= output_np, ax= ax)
plt.title('Output')

# ax = fig.add_subplot(r, c, 4)
# plt.imshow(image)

savefig(plt, "images/3d_conv.png")
plt.show()
plt.close()