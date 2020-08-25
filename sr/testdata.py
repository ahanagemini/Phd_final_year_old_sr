#!/usr/bin/env python3
import sys

if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("Usage: testdata.py data-directory-path")
        sys.exit(1)
    fpath = sys.argv[1]
