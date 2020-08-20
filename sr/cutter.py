#!/usr/bin/env python3
"""Usage: cutter.py
cutter.py --input-directory=IDIR --output-directory=ODIR

--input-directory=IDIR  Some directory [default: ./data]
--output-directory=ODIR  Some directory [default: ./mdata]

Example: python3.8 sr/cutter.py --input-directory=idata --output-directory=mdata
"""
import os
import json
from docopt import docopt
from pathlib import Path
import numpy as np


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
        "lower_quartile": lower_quartile
    }


def process(ifile, ofile):
    """
    ifile: input file path for matrix
    ofile: use this to chop input into pieces and write out
    """
    print("Processing: ", ifile)
    imatrix = np.load(ifile)
    imatrix = imatrix.f.arr_0  # Load data from inside file.
    stats = computestats(imatrix)
    if stats["upper_quartile"] - stats["lower_quartile"] < 0.01:
        continue
    prefix = ofile.stem
    odir = ofile.parent
    os.makedirs(odir)
    with open(odir / "stats.json", "w") as outfile:
        json.dump(stats, outfile)
    # mlist = matrixcutter(m)
    # Fill this direcory up with prefix_xx_xx.npz files.


def scan_idir(ipath, opath):
    """
    Returns (x,y) pairs so that x can be processed to create y
    """
    return [
        (x, opath / str(x)[len(str(ipath)) + 1 :]) for x in sorted(ipath.rglob("*.npz"))
    ]


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
