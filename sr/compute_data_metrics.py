#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 18:33:58 2020

@author: venkat
"""

import os
import math
from pathlib import Path
import numpy as np
import json

train_dir = Path("/home/ahana/CutterOutput/train")
val_dir = Path("/home/ahana/CutterOutput/valid")
test_dir = Path("/home/ahana/CutterOutput/test")

extensions = ["*.npz", "*.npy", "*.png", "*.tif", "*.jpeg", "*.jpg", "*.gif"]
files_list = []
[files_list.extend(train_dir.rglob(x)) for x in extensions]
sum = 0
var = 0
for i, files in enumerate(files_list):
    file_folder_name = os.path.splitext(files.name)[0]
    image = np.load(files)
    mat = image.f.arr_0
    sum = sum + np.mean(mat)

data_mean = sum / len(files_list)
print(data_mean)
for i, files in enumerate(files_list):
    file_folder_name = os.path.splitext(files.name)[0]
    image = np.load(files)
    mat = image.f.arr_0
    var = var + np.mean(np.square(mat - data_mean))

var = var / len(files_list)
stddev = math.sqrt(var)

stats = {
        "mean": data_mean,
        "std": stddev,
        "max_val": np.iinfo(mat.dtype).max
    }
print(stats)

for directories in os.listdir(train_dir):
    if os.path.isdir(os.path.join(train_dir, directories)):
        with open(os.path.join(train_dir, directories)+'/stat_global.json', 'w') as f:
            json.dump(stats, f)
for directories in os.listdir(val_dir):
    if os.path.isdir(os.path.join(val_dir, directories)):
        with open(os.path.join(val_dir, directories)+'/stat_global.json', 'w') as f:
            json.dump(stats, f)
for directories in os.listdir(test_dir):
    if os.path.isdir(os.path.join(test_dir, directories)):
        with open(os.path.join(test_dir, directories)+'/stat_global.json', 'w') as f:
            json.dump(stats, f)
