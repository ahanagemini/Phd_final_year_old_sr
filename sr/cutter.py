#!/usr/bin/env python3
"""Usage: cutter.py 
cutter.py --input-directory=IDIR --output-directory=ODIR

--input-directory=IDIR  Some directory [default: ./data]
--output-directory=ODIR  Some directory [default: ./mdata]

"""
from docopt import docopt
from pathlib import Path



def scan_idir(ipath):
    return sorted(ipath.rglob("*.npz"))

def main():
    arguments = docopt(__doc__, version='Matrix cutter system')
    idir = Path(arguments['--input-directory'])
    odir = Path(arguments['--output-directory'])
    L = scan_idir(idir)
    print(len(L))

if __name__ == '__main__':
    main()