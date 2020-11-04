#!/usr/bin/env python3

"""Usage: cutter.py --input-directory=IDIR --output-directory=ODIR --percentage=100 --patch_size=256 \
                    --train_size=0.9 --val_size=0.05 [--no_test_patch]
          cutter.py --help | -help | -h

--input-directory=IDIR  Some directory [default: ./data]
--output-directory=ODIR  Some directory [default: ./mdata]
--percentage=100 percentage of data to process [default: 100]
--patch_size=256 the image patch size is width, height [default: 256]
--train_size=0.9  the percentage of files in the train set
--val_size=0.05 the percentage of files in the validation set
--no_test_patch  No breaking the test set image sinto patches

cutter expects the input directory of images to be of the following structure.
Input_directory->patient_folder->patient_image.
The Output directory will be as follows Output_directory->train/valid/test->
patient_folder->patient_image and stats.jsonfile

Example: python3.8 sr/cutter.py --input-directory=idata --output-directory=mdata --percentage=100 --patch_size=256

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
import nibabel as nib


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

    ImagePaths = [".jpg", ".png", ".jpeg", ".gif"]
    ImageArrayPaths = [".npy", ".npz"]
    TiffPaths = [".tiff", ".tif"]
    niftiPaths = [".nii", ".nii.gz"]
    fname = str(ifile.name).lower()
    fileExt = "." + ".".join(fname.split(".")[1:])
    if fileExt in ImagePaths:
        image = Image.open(ifile)
        image = np.array(image.convert(mode="L"))
    if fileExt in TiffPaths:
        # 16 bit tifffiles are not read correctly by Pillow
        image = tifffile.imread(str(ifile))
    if fileExt == ".npz":
        image = np.load(ifile)
        image = image.f.arr_0  # Load data from inside file.
    elif fileExt in ImageArrayPaths:
        image = np.load(ifile)
    elif fileExt in niftiPaths:
        img = nib.load(ifile)
        image = img.get_data()

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
    if img_height <= height and img_width <= width:
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


def matrix_dictionary_update(
    key_matrix_map, matrix_key_map, file_names_map, imatrix, key, ofile, file_name
):
    if imatrix.ndim == 2:
        key_matrix_map.append((imatrix, key))
        matrix_key_map[key] = ofile
        file_names_map[key] = file_name
        key = key + 1
    else:
        # slicing in axial orientation for 3D data
        if "Sag_" in file_name:
            # slicing to get axial slices from sagitally collected data
            for i in range(imatrix.shape[1]):
                slice_i = imatrix[:,i,:]
                key_matrix_map.append((slice_i, key))
                matrix_key_map[key] = ofile
                file_names_map[key] = f"{file_name}_{i}"
                key = key + 1
        else:
            # Slicing to get axial slices from axially collected data
            for i in range(imatrix.shape[2]):
                slice_i = imatrix[:,:,i]
                key_matrix_map.append((slice_i, key))
                matrix_key_map[key] = ofile
                file_names_map[key] = f"{file_name}_{i}"
                key = key + 1

    return file_names_map, key_matrix_map, matrix_key_map, key


def process(L, stats_paths, train_size, val_size, patch_size, no_test_patch):
    """

    :param L: contains the input_file path and output_file path
    :param stats_paths: contains the paths for the stats json files
    :param train_size: train set size
    :param val_size: validation set size
    :param patch_size: this will cut the images to desired size
    :param no_test_patch: this ensures test set dtaa is not patched
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
    image_min = 0
    image_max = 0
    train_len = int(train_size * len(L))
    test_start = int(train_size * len(L)) + int(val_size * len(L))
    for i, (ifile, ofile) in enumerate(L):
        # print("processing" + str(ifile), end=" ")
        imatrix = loader(ifile)
        # file_name = os.path.splitext(ifile.name)[0]
        file_name = str(ifile.name).split(".")[0]
        #if imatrix.shape[0] < patch_size or imatrix.shape[1] < patch_size:
        #    print("Skipping because file is constant or size is too small.")
        #    continue

        matrix_vector = np.asarray(imatrix).reshape(-1)
        if i < train_len:
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
        # Keep track of key at which test set starts 
        if i == test_start:
            test_start_key = key
        file_names_map, key_matrix_map, matrix_key_map, key = matrix_dictionary_update(
            key_matrix_map,
            matrix_key_map,
            file_names_map,
            imatrix,
            key,
            ofile,
            file_name,
        )
        # print("\n")
    
    total_mean = total_sum / total_count
    total_variance = (total_square_sum / total_count) - (total_sum / total_count) ** 2
    stats = {}
    stats["mean"] = total_mean
    stats["variance"] = total_variance
    stats["std"] = np.sqrt(total_variance)
    stats["max"] = float(image_max)
    stats["min"] = float(image_min)
    width, height = patch_size, patch_size
    print("start file creation")
    for i in range(len(key_matrix_map)):
        matrix, key = key_matrix_map[i]
        opath = matrix_key_map[key]
        prefix = file_names_map[key]
        odir = opath
        if not os.path.isdir(odir):
            os.makedirs(odir)
        # if this is part of test set and no_test_patch is set,
        # Prevent patching of test set image
        if i >= test_start_key and no_test_patch:
            mlist = matrix_cutter(matrix, height=matrix.shape[0],
                                  width=matrix.shape[1])
        else:
            mlist = matrix_cutter(matrix, height=height, width=width)
        for i, j, mat in mlist:
            fname = str(prefix) + "_" + str(i) + "_" + str(j)
            np.savez_compressed(odir / fname, mat)
    # saving stats file in train directory
    for i in range(len(stats_paths)):
        with open(stats_paths[i] / "stats.json", "w") as outfile:
            json.dump(stats, outfile)

    print("Done")
    del (
        image_max,
        image_min,
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
        width,
        height,
    )


def scan_idir(ipath, opath, percentage, train_size=0.9, valid_size=0.05):
    """
    Returns (x,y) pairs so that x can be processed to create y
            train length to decide the images over which stats are computed
    """
    extensions = [
        "*.npy",
        "*.npz",
        "*.png",
        "*.jpg",
        "*.gif",
        "*.tif",
        "*.jpeg",
        "*.tiff",
        "*.nii",
        "*.nii.gz"
    ]
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
            input_folder = Path(input_folder)
            # if we have paired data for LR and HE,
            # we randomly shuffle to ensure order of paired data is maintained
            # and then treat LR and HR data as 2 separate distributions
            if set(os.listdir(input_folder)) == set(["LR", "HR"]):
                print("Found separate LR and HR directory. Creating paired dataset")
                folders_list.append(folder_name + "/LR")
                folders_list.append(folder_name + "/HR")
                [folders_files.extend(input_folder.rglob(x)) for x in extensions]
                sorted(folders_files)
                num_pairs = len(folders_files) // 2
                paired_files = [(folders_files[i],
                                 folders_files[num_pairs + i]) for i in range(num_pairs)]
                size = int(len(paired_files) * (percentage / 100))
                random.Random(4).shuffle(paired_files)
                paired_files = paired_files[:size]
                folder_file_map[folder_name+"/HR"] = [elem[0] for elem in paired_files]
                folder_file_map[folder_name+"/LR"] = [elem[1] for elem in paired_files]
            else:
                folders_list.append(folder_name)
                [folders_files.extend(input_folder.rglob(x)) for x in extensions]
                size = int(len(folders_files) * (percentage / 100))
                random.Random(4).shuffle(folders_files)
                folders_files = folders_files[:size]
                folder_file_map[folder_name] = folders_files
        folders_files = []
    paths = ["train", "valid", "test"]
    folder_input_output_map = {}
    for folder in folders_list:
        L = []
        stats_paths = []
        folder_files = folder_file_map[folder]
        for i, files in enumerate(folder_files):
            if i < int(train_size * len(folder_files)):
                L.append((files, opath / paths[0] / folder))
                stats_paths.append(Path(opath / paths[0] / folder))

            elif i >= int(train_size * len(folder_files)) and i < int(
                (train_size + valid_size) * len(folder_files)
            ):
                L.append((files, opath / paths[1] / folder))
                stats_paths.append(Path(opath / paths[1] / folder))
            else:
                L.append((files, opath / paths[2] / folder))
                stats_paths.append(Path(opath / paths[2] / folder))
        folder_input_output_map[folder] = L, stats_paths

    return folder_input_output_map, train_size


def main():
    """
    process every file individually here
    """
    arguments = docopt(__doc__, version="Matrix cutter system")
    idir = Path(arguments["--input-directory"])
    odir = Path(arguments["--output-directory"])
    percentage = int(arguments["--percentage"])
    patch_size = int(arguments["--patch_size"])
    train_size = float(arguments["--train_size"])
    val_size = float(arguments["--val_size"])
    no_test_patch = arguments["--no_test_patch"]
    assert not odir.is_dir(), "Please provide a non-existent output directory!"
    folder_map, train_size = scan_idir(idir, odir, percentage,
                                       train_size, val_size)
    print(folder_map.keys())
    for folder in folder_map.keys():
        print(folder)
        L, stats_paths = folder_map[folder]
        process(L, stats_paths, train_size, val_size, patch_size, no_test_patch)


if __name__ == "__main__":
    main()
