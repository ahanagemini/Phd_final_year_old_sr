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

for files in os.scandir(input_file):
    file_folder_name = os.path.splitext(files.name)[0]
    file_name = file_folder_name + ".npy"
    image = Image.open(files.path)
    image = image.convert(mode="L")
    os.makedirs(output_file / file_folder_name)
    np.save(output_file/file_folder_name/file_name, np.array(image))