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

from cutter import loader
import numpy as np
import tifffile
import json



def process(infile, outfile):
    '''

    :param infile: Input files directory
    :param outfile: Ouput files directory
    :return:
    '''
    extensions = ["*.npz", "*.npy", "*.png", "*.tif", "*.jpeg", "*.jpg", "*.gif"]
    assert_stats(infile)
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
        image = image.convert(mode="F")
        output_file = outfile
        if not output_file.is_dir():
            os.makedirs(output_file)
        np.savez_compressed(output_file / file_name, np.array(image))

def stat_calculator(input_path):
    print("creating stats")
    total_sum = 0.0
    total_square_sum = 0.0
    total_count = 0.0
    image_min = 0
    image_max = 0
    for image_path in input_path.rglob("*.npz"):
        imatrix = loader(image_path)
        matrix_vector = np.asarray(imatrix).reshape(-1)
        square_vector = np.square(matrix_vector, dtype=np.float)
        matrix_sum = np.sum(matrix_vector, dtype=np.float)
        square_sum = np.sum(square_vector, dtype=np.float)
        matrix_count = len(matrix_vector)

        # this information is for total mean calculation
        total_sum = total_sum + matrix_sum
        total_square_sum = total_square_sum + square_sum
        total_count = total_count + matrix_count

        # maximum and minimum
        matrix_max = np.max(matrix_vector)
        matrix_min = np.min(matrix_vector)
        if image_max < matrix_max:
            image_max = matrix_max

        if image_min > matrix_min:
            image_min = matrix_min
    if total_count == 0:
        print("no elements loaded creating stats failed")
        return
    total_mean = total_sum / total_count
    total_variance = (total_square_sum / total_count) - (total_sum / total_count) ** 2
    stats = {}
    stats["mean"] = total_mean
    stats["std"] = np.sqrt(total_variance)
    stats["max"] = float(image_max)
    stats["min"] = float(image_min)

    return stats

def assert_stats(input_directory):
    """
    Returns stats. If stats.json is not present, computes it.
    """
    #input_directory = Path(conf.input_dir_path)
    if not os.path.isfile(str(input_directory / "stats.json")):
        """ calculate stats"""
        stats = stat_calculator(input_directory)

        with open(str(input_directory / "stats.json"), "w") as sfile:
            json.dump(stats, sfile)

    else:
        print("loading available stats")
        stats = json.load(open(str(input_directory / "stats.json")))
    return stats

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
