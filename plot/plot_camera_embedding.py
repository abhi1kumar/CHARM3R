"""
    Sample Run:
    python .py
"""
import os, sys

import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

import numpy as np
np.set_printoptions   (precision= 2, suppress= True)

from plot.common_operations import *
import plot.plotting_params as params
import matplotlib

from lib.helpers.homography_helper import get_intrinsics_from_fov

h    = w = 512
fov  = 60
U, V = np.meshgrid(np.arange(w), np.arange(h))
r = 2
c = 2
cmap1 = 'magma_r'

intrinsics = get_intrinsics_from_fov(w, h, fov_degree= 64.5615)
fx = intrinsics[0, 0]
fy = intrinsics[1, 1]
u0 = intrinsics[0, 2]
v0 = intrinsics[1, 2]

EPS = 1e-4

outU1 = (U - u0)/u0
outV1 = (V - v0)/v0
outU2 = fx/(U - u0 + EPS)
outV2 = fy/(V - v0 + EPS)

scale = 5

plt.figure(figsize=(12,12), dpi= params.DPI)
plt.subplot(r,c,1)
plt.imshow(outU1, cmap= cmap1, vmin= -1, vmax= 1)
plt.title('Vanilla X ' + r'$(u-u_0)$')
plt.axis('off')
plt.subplot(r,c,2)
plt.imshow(outU2, cmap= cmap1, vmin= -scale, vmax= scale)
plt.title('Inverted X ' + r'$1/(u-u_0)$')
plt.axis('off')
plt.subplot(r,c,3)
plt.imshow(outV1, cmap= cmap1, vmin= -1, vmax= 1)
plt.title('Vanilla Y '  + r'$(v-v_0)$')
plt.axis('off')
plt.subplot(r,c,4)
plt.imshow(outV2, cmap= cmap1, vmin= -scale, vmax= scale)
plt.title('Inverted Y ' + r'$1/(v-v_0)$')
plt.axis('off')
savefig(plt, "images/camera_embeddings.png")
plt.show()
plt.close()