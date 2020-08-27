#!/usr/bin/env python3
import sys
from pathlib import Path
import os

def fileChecker(file_path):
    file_extensions = [".npy", ".npz", ".png", ".jpg", ".gif", ".jpeg"]
    for image_file in file_path.rglob("*.*"):
        if image_file.is_file():
            image_file_ext = os.path.splitext(image_file.name)[1].lower()
            if image_file_ext not in file_extensions:
                assert False



def directoryChecker(directory_path):
    directory_names = ["train", "test", "validate"]
    for folder in os.scandir(directory_path):
        if folder.is_dir():
            if folder.name not in directory_names:
                assert False
            else:
                fileChecker(directory_path / folder.name)

if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("Usage: testdata.py data-directory-path")
        sys.exit(1)
    fpath = Path(sys.argv[1])
    directoryChecker(fpath)



