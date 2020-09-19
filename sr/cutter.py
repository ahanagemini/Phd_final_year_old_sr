#!/usr/bin/env python3

"""Usage: cutter.py --input-directory=IDIR --output-directory=ODIR
          cutter.py --help | -help | -h

--input-directory=IDIR  Some directory [default: ./data]
--output-directory=ODIR  Some directory [default: ./mdata]

cutter expects the input directory of images to be of the following structure.
Input_directory->patient_folder->patient_image.
The Output directory will be as follows Output_directory->train/valid/test->
patient_folder->patient_image and stats.jsonfile

Example: python3.8 sr/cutter.py --input-directory=idata --output-directory=mdata

Options:
--h | -help | --help
"""

import os
import json
from PIL import Image
import tifffile
from docopt import docopt
from pathlib import Path
import numpy as np


def loader(ifile):
    """


    Parameters
    ----------
    file : TYPE
        DESCRIPTION.

    Returns
    -------
    imageArray : TYPE
        DESCRIPTION.

    """

    image_paths = [".jpg", ".png", ".jpeg", ".gif", ".tif"]
    image_array_paths = [".npy", ".npz"]
    file_ext = os.path.splitext(ifile.name)[1].lower()
    if file_ext in image_paths:
        image = Image.open(ifile)
        image = np.array(image.convert(mode="L"))
    if file_ext == ".tiff":
        image = tifffile.imread(ifile)
    if file_ext == ".npz":
        image = np.load(ifile)
        image = image.f.arr_0  # Load data from inside file.
    elif file_ext in image_array_paths:
        image = np.load(ifile)
    return image


def matrix_cutter(img, width=256, height=256):
    """


    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    height : TYPE, optional
        DESCRIPTION. The default is 256.
    width : TYPE, optional
        DESCRIPTION. The default is 256.

    Returns
    -------
    None.

    """
    images = []
    img_height, img_width = img.shape

    # check if images have 256 width and 256 height if it does skip cutting
    if img_height == height and img_width == width:
        return [(0, 0, img)]

    for i, ih in enumerate(range(0, img_height, height)):
        for j, iw in enumerate(range(0, img_width, width)):
            posx = iw
            posy = ih
            if posx + width > img_width:
                posx = img_width - width
            if posy + height > img_height:
                posy = img_height - height

            cutimg = img[posy : posy + height, posx : posx + width]
            cutimg_height, cutimg_width = cutimg.shape
            assert cutimg_height == height and cutimg_width == width
            images.append((i, j, cutimg))
    return images


def process(L, stats_path):
    """

    :param L: contains the input_file path and output_file path
    :param stats_path: contains the path for the stats json file
    :return:
    """
    matrices = []
    file_names_map = {}
    key_matrix_map = []
    matrix_key_map = {}
    key = 0
    total_sum = 0.0
    total_square_sum = 0.0
    total_count = 0.0
    maxval = 0
    minval = 0
    for k, (ifile, ofile) in enumerate(L):
        print("processing" + str(ifile), end=" ")
        imatrix = loader(ifile)
        file_name = os.path.splitext(ifile.name)[0]
        if imatrix.shape[0] < 256 or imatrix.shape[1] < 256:
            print("Skipping because file is constant or size is too small.")
            continue
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
        maxval = max(maxval, matrix_max)
        minval = min(minval, matrix_min)

        key_matrix_map.append((imatrix, key))
        matrix_key_map[key] = ofile
        file_names_map[key] = file_name
        key = key + 1
        print("\n")

    total_mean = total_sum / total_count
    total_variance = (total_square_sum / total_count) - (total_sum / total_count) ** 2
    stats = {}
    stats["mean"] = total_mean
    stats["variance"] = total_variance
    stats["std"] = np.sqrt(total_variance)
    stats["max"] = float(maxval)
    stats["min"] = float(minval)

    print("start file creation")
    for i in range(len(key_matrix_map)):
        matrix, key = key_matrix_map[i]
        opath = matrix_key_map[key]
        prefix = file_names_map[key]
        odir = opath
        if not os.path.isdir(odir):
            os.makedirs(odir)
        mlist = matrix_cutter(matrix)
        for k, j, mat in mlist:
            fname = str(prefix) + "_" + str(k) + "_" + str(j)
            np.savez_compressed(odir / fname, mat)
    # saving stats file in train directory
    with open(stats_path / "stats.json", "w") as outfile:
        json.dump(stats, outfile)

    print("Done")
    del (
        maxval,
        minval,
        matrices,
        file_names_map,
        key_matrix_map,
        matrix_key_map,
        key,
        stats,
        total_sum,
        total_count,
        total_mean,
        total_variance,
    )


def scan_idir(ipath, opath, train_size=0.9, valid_size=0.05):
    """
    Returns (x,y) pairs so that x can be processed to create y
    """
    extensions = ["*.npy", "*.npz", "*.png", "*.jpg", "*.gif", "*.tif", "*.jpeg"]
    folders_list = []
    folders_files = []
    folder_file_map = {}
    if train_size + valid_size > 1.0:
        print("THe train_size and valid_size is invalid")
        return
    if train_size + valid_size == 1.0:
        print("There will be no testing files")

    folders = os.scandir(ipath)
    for input_folder in folders:
        if input_folder.is_dir():
            folder_name = input_folder.name
            folders_list.append(folder_name)
            input_folder = Path(input_folder)
            [folders_files.extend(input_folder.rglob(x)) for x in extensions]
            folder_file_map[folder_name] = folders_files
        folders_files = []
    paths = ["train", "test", "valid"]
    folder_input_output_map = {}
    stats_path = Path("")
    for folder in folders_list:
        L = []
        folder_files = folder_file_map[folder]
        for i, files in enumerate(folder_files):
            if i < int(train_size * len(folder_files)):
                L.append((files, opath / paths[0] / folder))
                stats_path = Path(opath / paths[0] / folder)

            elif i >= int(train_size * len(folder_files)) and i < int(
                (train_size + valid_size) * len(folder_files)
            ):
                L.append((files, opath / paths[1] / folder))
            else:
                L.append((files, opath / paths[2] / folder))
        folder_input_output_map[folder] = L, stats_path

    return folder_input_output_map


def main():
    """
    process every file individually here
    """
    arguments = docopt(__doc__, version="Matrix cutter system")
    idir = Path(arguments["--input-directory"])
    odir = Path(arguments["--output-directory"])
    assert not odir.is_dir(), "Please provide a non-existent output directory!"
    folder_map = scan_idir(idir, odir)

    for folder in folder_map.keys():
        L, stats_path = folder_map[folder]
        process(L, stats_path)


if __name__ == "__main__":
    main()
