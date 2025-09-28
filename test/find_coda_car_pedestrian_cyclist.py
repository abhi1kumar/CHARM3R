"""
    Sample Run:
    python .py
"""
import os, sys
sys.path.append(os.getcwd())

import numpy as np
import glob
np.set_printoptions   (precision= 2, suppress= True)

from lib.helpers.rpn_util import filter_boxes_by_cars
from lib.helpers.file_io  import write_lines, read_csv

input_folder   = "data/coda/training/label_2/"
list_of_labels = sorted(glob.glob(input_folder + "/*.txt"))

output_path    = "data/coda/ImageSets/val2.txt"
lines_with_return_character = []

relevant_cnt = 0
for i, label_path in enumerate(list_of_labels):
    boxes = read_csv(label_path, use_pandas= True)
    img_id = os.path.basename(label_path).replace(".txt", "")
    if boxes is not None:
        mask1 = np.logical_or(boxes[:, 0] == "Car"       , boxes[:, 0] == "car")
        mask2 = np.logical_or(boxes[:, 0] == "Pedestrian", boxes[:, 0] == "Cyclist")
        mask  = np.logical_or(mask1, mask2)
        if np.any(mask):
            relevant_cnt += 1
            lines_with_return_character += [img_id + "\n"]

    if (i+1) % 5000 == 0 or (i+1) == len(list_of_labels):
        print("{:5d}/{:5d} images done".format(relevant_cnt, i+1))

write_lines(output_path, lines_with_return_character)
