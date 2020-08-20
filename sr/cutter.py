#!/usr/bin/env python3
"""Usage: cutter.py 
cutter.py --input-directory=IDIR --output-directory=ODIR

--input-directory=IDIR  Some directory [default: ./data]
--output-directory=ODIR  Some directory [default: ./mdata]

Example: python3.8 sr/cutter.py --input-directory=idata --output-directory=mdata
"""
import os
from docopt import docopt
from pathlib import Path
import numpy as np


def computestats(m):
    upper_quartile = np.percentile(m, 90)
    lower_quartile = np.percentile(m, 10)
    return {
        "mean": np.mean(m),
        "std": np.std(m),
        "upper_quartile": upper_quartile,
        "lower_quartile": lower_quartile,
    }


def process(ifile, ofile):
    print("Processing: ", ifile)
    m = np.load(ifile)
    m = m.f.arr_0  # Load data from inside file.
    stats = computestats(m)
    prefix = ofile.stem
    odir = ofile.parent
    os.makedirs(odir)
    # mlist = matrixcutter(m)
    # Fill this direcory up with prefix_xx_xx.npz files.


def scan_idir(ipath, opath):
    return [
        (x, opath / str(x)[len(str(ipath)) + 1 :]) for x in sorted(ipath.rglob("*.npz"))
    ]


def main():
    arguments = docopt(__doc__, version="Matrix cutter system")
    idir = Path(arguments["--input-directory"])
    odir = Path(arguments["--output-directory"])
    assert not odir.is_dir(), "Please provide a non-existent output directory!"
    L = scan_idir(idir, odir)
    for inpfile, outfile in L:
        process(inpfile, outfile)


if __name__ == "__main__":
    main()
