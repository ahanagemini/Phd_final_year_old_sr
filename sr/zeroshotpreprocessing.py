#!/usr/bin/env python3

"""Usage:   zeroshotpreprocessing.py --input-directory=input_path --output-directory=output_path --samples=samples
            zeroshotpreprocessing.py --help | -help | -h

This will calulate the kernel using KernelGAN and using this kernel will calculate patches and all these data will be
saved in the output directory
Arguments:
  input-directory: input files
  output-directory: output location
  samples: no. samples required with scale factor of 5%
Options:
  -h --help -h
"""

from cutter import loader
from kernelgan.imresize import imresize
from kernelgan.train import train
import numpy as np
import os
from pathlib import Path
from docopt import docopt
import json


def image_stat_processing(input_directory, output_directory, samples):
    """

    Parameters
    ----------
    param input_directory: input location
    output_directory: output location
    samples: no. of downscaling samples by 5% using kernel

    Returns
    -------

    """
    image_path_dict = {}
    for image_path in input_directory.rglob("*.npz"):
        image_parent = image_path.parent
        if image_parent not in image_path_dict.keys():
            image_path_dict[image_parent] = []
        image = loader(image_path)
        image_path_dict[image_parent].append(image)

    for _, image_parent in enumerate(image_path_dict):
        stats = json.load(image_parent / "stats.json", "w")
        if not os.path.isdir(output_directory):
            os.makedirs(str(output_directory / image_parent))
        for image_file in image_path_dict[image_parent]:
            image_name = os.path.splitext(image_file.name)[0]
            image = loader(image_file)
            kernel = train(image, stats)
            sample_list = []
            for _ in range(samples):
                out_image = imresize(im=image, scale_factor=0.95, kernel=kernel)
                sample_list.append(out_image)
                image = out_image
            for i, image_patch in enumerate(sample_list):
                np.savez_compressed(
                    str(output_directory / image_parent / image_name + str(i)),
                    image_patch,
                )
            np.savez_compressed(
                str(output_directory / image_parent / "kernel"), image_patch
            )
        with open(str(output_directory / image_parent / "stats.json"), "w") as sfile:
            json.dump(stats, sfile)


def process(p_args):
    """

    Parameters
    ----------
    param arguments: this contains arguments taken from terminal

    Returns
    -------

    """
    input_directory = Path(p_args["--input-directory"])
    output_directory = Path(p_args["--output-directory"])
    samples = Path(p_args["--samples"])
    image_stat_processing(input_directory, output_directory, samples)


if __name__ == "__main__":
    arguments = docopt(__doc__)
    process(arguments)
