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
import random


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

    ImagePaths = [".jpg", ".png", ".jpeg", ".gif", ".tif"]
    ImageArrayPaths = [".npy", ".npz"]
    fileExt = os.path.splitext(ifile.name)[1].lower()
    if fileExt in ImagePaths:
        image = Image.open(ifile)
        image = np.array(image.convert(mode="L"))
    if fileExt == ".tiff":
        image = tifffile.imread(ifile)
    if fileExt == ".npz":
        image = np.load(ifile)
        image = image.f.arr_0  # Load data from inside file.
    elif fileExt in ImageArrayPaths:
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


def computestats(imatrix):
    """
    Compute basic statistics of the loaded matrix
    """
    upper_quartile = float(np.percentile(imatrix, 90))
    lower_quartile = float(np.percentile(imatrix, 10))
    return {
        "mean": float(np.mean(imatrix)),
        "std": float(np.std(imatrix)),
        "upper_quartile": upper_quartile,
        "lower_quartile": lower_quartile,
    }


def process(ifile, ofile):
    """
    ifile: input file path for matrix
    ofile: use this to chop input into pieces and write out
    """
    print("Processing: ", ifile, "...", end="")
    imatrix = loader(ifile)
    # imatrix = np.load(ifile)
    # imatrix = imatrix.f.arr_0  # Load data from inside file.
    stats = computestats(imatrix)
    prefix = ofile.stem
    odir = ofile
    if not odir.exists():
        os.makedirs(odir)
    with open(odir / "stats.json", "w") as outfile:
        json.dump(stats, outfile)

    if (
        (stats["upper_quartile"] - stats["lower_quartile"] < 0.01)
        or imatrix.shape[0] < 256
        or imatrix.shape[1] < 256
    ):

        print("Skipping because file is constant or size is too small.")
        return

    mlist = matrix_cutter(imatrix)
    for i, j, mat in mlist:
        fname = str(prefix) + "_" + str(i) + "_" + str(j)
        np.savez_compressed(odir / fname, mat)

    # Fill this direcory up with prefix_xx_xx.npz files.
    print("Done")


def scan_idir(ipath, opath, train_size=0.9, valid_size=0.05):
    """
    Returns (x,y) pairs so that x can be processed to create y
    """
    extensions = ["*.npy", "*.npz", "*.png", "*.jpg", "*.gif", "*.tif", "*.jpeg"]
    folders_list = []
    files_list = []
    folders_files = []
    folder_file_map = {}
    if train_size + valid_size > 1.0:
        print("THe train_size and valid_size is invalid")
        return
    if train_size + valid_size == 1.0:
        print("There will be no testing files")

    [files_list.extend(ipath.rglob(x)) for x in extensions]
    for input_file in files_list:
        folder_name = input_file.parent.name
        if folder_name in folders_list:
            continue
        folders_list.append(folder_name)
        [folders_files.extend(input_file.parent.rglob(x)) for x in extensions]
        folder_file_map[folder_name] = folders_files
        folders_files = []

    random.shuffle(folders_list)
    L = []
    paths = ["train", "test", "valid"]

    if len(folders_list) == 1:
        folders_files = folder_file_map[folders_list[0]]
        for i, files in enumerate(folders_files):
            file_name = os.path.splitext(files.name)[0]
            if i < int(train_size * len(folders_list)):
                L.append((files, opath / paths[0] / file_name))
            elif i >= int(train_size * len(folders_list)) and i < int(
                (train_size + valid_size) * len(folders_list)
            ):
                L.append((files, opath / paths[1] / file_name))
            else:
                L.append((files, opath / paths[2] / file_name))

    else:
        for i, x in enumerate(folders_list):
            if i < int(train_size * len(folders_list)):
                folders_files = folder_file_map[folders_list[i]]
                for files in folders_files:
                    file_name = os.path.splitext(files.name)
                    L.append((files, opath / paths[0] / x / file_name))
            elif i >= int(train_size * len(folders_list)) and i < int(
                (train_size + valid_size) * len(folders_list)
            ):
                folders_files = folder_file_map[folders_list[i]]
                for files in folders_files:
                    file_name = os.path.splitext(files.name)
                    L.append((files, opath / paths[1] / x / file_name))
            else:
                folders_files = folder_file_map[folders_list[i]]
                for files in folders_files:
                    file_name = os.path.splitext(files.name)
                    L.append((files, opath / paths[2] / x / file_name))
    return L


def main():
    """
    process every file individually here
    """
    arguments = docopt(__doc__, version="Matrix cutter system")
    idir = Path(arguments["--input-directory"]).resolve()
    odir = Path(arguments["--output-directory"]).resolve()
    assert not odir.is_dir(), "Please provide a non-existent output directory!"
    L = scan_idir(idir, odir)
    for inpfile, outfile in L:
        process(inpfile, outfile)


if __name__ == "__main__":
    main()
