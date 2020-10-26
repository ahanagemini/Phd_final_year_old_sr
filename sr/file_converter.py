#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
usage:   
file_converter.py --input_files=input_files_path --output_files=output_files_path
file_converter.py --help | -help | -h

Arguments:
  input_files_path: Path with directory structure input_directory->ImageFiles
  output_files_path: Output Path with desired structure input_directory->Image_file_name->ImageFiles
Options:
  -h --help -h
"""

"""
Created on Thu Aug 27 18:33:58 2020

@author: venkat

This file can be used to form directory structure required for executing cutter
if the given directory has only image files. For example cutter requires directory
to be of the form Input_directory->Patient->PatientImage. This file converts,
Input_directory->Image_file to Input_directory->Image_file_name->Image_file

"""

import os
from pathlib import Path
from docopt import docopt
from PIL import Image
from tqdm import tqdm

import numpy as np
import tifffile



def process(infile, outfile):
    '''

    :param infile: Input files directory
    :param outfile: Ouput files directory
    :return:
    '''
    extensions = ["*.npz", "*.npy", "*.png", "*.tif", "*.jpeg", "*.jpg", "*.gif"]
    files_list = []
    [files_list.extend(infile.rglob(x)) for x in extensions]
    for i, files in tqdm(enumerate(files_list), total=len(files_list)):
        image_paths = [".png", ".jpg", ".jpeg", ".gif", ".tif"]
        file_folder_name, file_ext = (
            os.path.splitext(files.name)[0],
            os.path.splitext(files.name)[1],
        )
        file_name = file_folder_name
        if file_ext == ".npy":
            image = Image.fromarray(np.load(files))
        elif file_ext == ".npz":
            image = np.load(files)
            image = Image.fromarray(image.f.arr_0)
        elif file_ext == ".tiff":
            image = Image.fromarray(tifffile.imread(files))
        elif file_ext in image_paths:
            image = Image.open(files)
        image = image.convert(mode="L")
        output_file = outfile
        if not output_file.is_dir():
            os.makedirs(output_file)
        np.savez_compressed(output_file / file_name, np.array(image))


def main():
    """
    This function changes the directory structure when given structure has only images
    """
    arguments = docopt(__doc__)
    idir = Path(arguments["--input_files"])
    odir = Path(arguments["--output_files"])
    process(idir, odir)


if __name__ == "__main__":
    main()
