#!/usr/bin/env python3
"""Usage: cutter.py 
cutter.py --input-directory=IDIR --output-directory=ODIR

--input-directory=IDIR  Some directory [default: ./data]
--output-directory=ODIR  Some directory [default: ./mdata]

"""
from docopt import docopt

def main():
    arguments = docopt(__doc__, version='Matrix cutter system')
    print(arguments)

if __name__ == '__main__':
    main()