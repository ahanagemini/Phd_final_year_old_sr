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
    assert directory_path.is_dir()
    directory_names = ["train", "test", "validate"]
    for folder in directory_path.rglob("*"):
        print("Folder:", folder.absolute())
        if folder.is_dir():
            if folder.name not in directory_names:
                assert False
            else:
                for dataset in folder.rglob("*"):
                    fileChecker(folder / dataset.name)


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: testdata.py data-directory-path")
        sys.exit(1)
    fpath = Path(sys.argv[1])
    directoryChecker(fpath)
