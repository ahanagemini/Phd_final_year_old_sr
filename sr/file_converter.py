#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 18:33:58 2020

@author: venkat
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np

input_file = Path(os.getcwd() + r"/DIV2K_train_HR/")

output_file = Path(os.getcwd() +r"/DIV2K_rev/")

extensions = ["*.npz", "*.npy", "*.png", "*.tif", "*.jpeg", "*.jpg", "*.gif"]
files_list = []
[files_list.extend(input_file.rglob(x)) for x in extensions]

for i, files in enumerate(files_list):
    file_folder_name = os.path.splitext(files.name)[0]
    file_name = file_folder_name
    image = Image.open(files.path)
    image = image.convert(mode="L")
    os.makedirs(output_file / file_folder_name)
    np.savez_compressed(output_file/file_folder_name/file_name, np.array(image))