#!/usr/bin/env python3
"""Usage: cutter.py
cutter.py --input-directory=IDIR --output-directory=ODIR

--input-directory=IDIR  Some directory [default: ./data]
--output-directory=ODIR  Some directory [default: ./mdata]

Example: python3.8 sr/cutter.py --input-directory=idata --output-directory=mdata
"""
import os
import json
from PIL import Image
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

    ImagePaths = [".jpg", ".png", ".jpeg", ".gif"]
    ImageArrayPaths = [".npy", ".npz"]
    fileExt = os.path.splitext(ifile.name)[1].lower()
    if fileExt in ImagePaths:
        image = np.array(Image.open(ifile))
    if fileExt == ".npz":
        image = np.load(ifile)
        image = image.f.arr_0  # Load data from inside file.
    elif fileExt in ImageArrayPaths:
        image = Image.np.load(ifile)
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
    img_height, img_width, img_channels= img.shape
    for i, ih in enumerate(range(0, img_height, height)):
        for j, iw in enumerate(range(0, img_width, width)):
            posx = iw
            posy = ih
            if posx + width > img_width:
                posx = img_width - width
            if posy + height > img_height:
                posy = img_height - height

            cutimg = img[posy : posy + height, posx : posx + width, :]
            cutimg_height, cutimg_width, cutimg_channels = cutimg.shape
            assert cutimg_height == height and cutimg_width == width and cutimg_channels == img_channels
            images.append((i, j, cutimg))
    return images


def computestats(imatrix):
    """
    Compute basic statistics of the loaded matrix
    """
    upper_quartile = np.percentile(imatrix, 90)
    lower_quartile = np.percentile(imatrix, 10)
    return {
        "mean": np.mean(imatrix),
        "std": np.std(imatrix),
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
    if (
        (stats["upper_quartile"] - stats["lower_quartile"] < 0.01)
        or imatrix.shape[0] < 256
        or imatrix.shape[1] < 256
    ):
        print("Skipping because file is constant or size is too small.")
        return
    prefix = ofile.stem
    odir = ofile
    os.makedirs(odir)
    with open(odir / "stats.json", "w") as outfile:
        json.dump(stats, outfile)

    mlist = matrix_cutter(imatrix)
    for i, j, mat in mlist:
        fname = str(prefix) + "_" + str(i) + "_" + str(j)
        np.savez_compressed(odir / fname, mat)

    # Fill this direcory up with prefix_xx_xx.npz files.
    print("Done")


def scan_idir(ipath, opath):
    """
    Returns (x,y) pairs so that x can be processed to create y
    """
    extensions = ["*.npy", "*.npz", "*.png", "*.jpg", "*.gif", "*.jpeg"]
    filesList = []

    [filesList.extend(sorted(ipath.rglob(x))) for x in extensions]

    L = []
    paths = ["Training", "Validation", "Testing"]
    for i, x in enumerate(filesList):
        if i < 0.9 * len(filesList):
            L.append((x, opath / paths[0] / "HR" / str(x)[len(str(ipath)) + 1:]))
        elif i >= 0.9 * len(filesList) and i < 0.95 * len(filesList):
            L.append((x, opath / paths[1] / "HR" / str(x)[len(str(ipath)) + 1:]))
        else:
            L.append((x, opath / paths[2] / "HR" / str(x)[len(str(ipath)) + 1:]))
    return L


def main():
    """
    process every file individually here
    """
    arguments = docopt(__doc__, version="Matrix cutter system")
    idir = Path(arguments["--input-directory"])
    odir = Path(arguments["--output-directory"])
    assert not odir.is_dir(), "Please provide a non-existent output directory!"
    L = scan_idir(idir, odir)
    for inpfile, outfile in L:
        process(inpfile, outfile)


if __name__ == "__main__":
    main()
