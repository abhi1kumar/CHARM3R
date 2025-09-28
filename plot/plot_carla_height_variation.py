"""
    Sample Run:
    python .py
"""
import os, sys
sys.path.append(os.getcwd())

import numpy as np
np.set_printoptions   (precision= 4, suppress= True)

from plot.common_operations import *
import plot.plotting_params as params
import matplotlib

import matplotlib.pyplot as plt

from lib.helpers.file_io import read_json, read_numpy, read_image

fs         = 32
matplotlib.rcParams.update({'font.size': fs})

heights_list = ['height-27', 'height-18', 'height0', 'height18', 'height30']
delta_H_list = ['-0.70', '-0.45' , '0', '+0.45', '+0.76']

heights_list = ['height-27', 'height-18', 'height-12', 'height0', 'height12', 'height18', 'height30']
delta_H_list = ['-0.70', '-0.45', '-0.30' , '0', '+0.30', '+0.45', '+0.76']

heights_list = ['height-27', 'height0', 'height30']
delta_H_list = ['-0.70', '0', '+0.76']

ilist = [4]#sorted(np.random.choice(17, 17, replace= False))
flist = [0] * len(ilist)
town = 'town03'

ilist = [1054, 1021, 1033, 1061, 1077, 1114, 1209, 1278]
flist = [9, 5, 3, 5, 3, 1, 6, 0]

ilist = [1054, 1061, 1114, 1278]
flist = [9, 5, 1, 0]

town = 'town05'

c = len(heights_list)

for a, (i, f) in enumerate(zip(ilist, flist)):
    key = str(i)
    f1 = str(f).zfill(4)
    plt.figure(figsize= (6*c,6), dpi= 75)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.02, hspace=None)

    for j, (height, dH) in enumerate(zip(heights_list, delta_H_list)):
        folder = os.path.join('data/carla/carla_abhinav/', height, town)
        image_path  = os.path.join(folder, key, 'image', f1 + '_00.jpg')
        image = read_image(image_path, rgb= True)
        
        plt.subplot(1,c,j+1)
        plt.imshow(image)
        plt.axis('off')
        if a == 0:
            if j == 0:
                title_text = "Bot"
            elif j == c//2:
                title_text = "Car"
            elif j == c-1:
                title_text = "Truck"
            else:
                title_text = ""
            title_text += "\n"
            title_text += r'$\Delta{}H\!=\!'+ dH + r'm$'
            plt.title(title_text)
            
   
    out_folder = os.path.join("images/height_variation", town)
    os.makedirs(out_folder, exist_ok= True)
    savefig(plt, os.path.join(out_folder, "{}_{}.png".format(key, f1)), newline= False)
