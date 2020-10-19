#!/usr/bin/env python3

"""Usage:   zeroshotpreprocessing.py --input-directory=input_path --output-directory=output_path --samples=samples
            zeroshotpreprocessing.py --help | -help | -h

This will calulate the kernel using KernelGAN and using this kernel will calculate patches and all these data will be
saved in the output directory

Arguments:
  --input-directory=input_path   : input files
  --output-directory=output_path : output location
  --samples=samples              : no. of scales generated for the input using a reduction factor of 5%

Options:
  -h --help -h

"""

from cutter import loader
from kernelgan import imresize
from kernelgan import train
import numpy as np
import os
from pathlib import Path
import json
from configs import Config


def image_stat_processing(conf):
    """

    Parameters
    ----------
    param input_directory: input location
    output_directory: output location
    samples: no. of downscaling samples by 5% using kernel

    Returns
    -------

    """
    conf.real_image = True
    output_directory = Path(conf.output_dir_path)
    image_path_dict = {}
    for image_path in Path(conf.input_dir_path).rglob("*.npz"):
        image_parent = image_path.parent
        if image_parent not in image_path_dict.keys():
            image_path_dict[image_parent] = []
        image_path_dict[image_parent].append(image_path)

    for _, image_parent in enumerate(image_path_dict):
        stats = json.load(open(str(image_parent / "stats.json")))
        if not os.path.isdir(output_directory):
            os.makedirs(str(output_directory / image_parent.name))
        for image_file in image_path_dict[image_parent]:
            image_name = os.path.splitext(image_file.name)[0]
            image = loader(image_file)
            conf.image = image
            conf.stats = stats
            print(image)
            print(stats)
            kernel = train(conf)
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


if __name__ == "__main__":
    conf = Config().parse()
    print(type(conf))
    conf.x = 1000
    from pprint import pprint

    pprint(conf)
    print("samples = ", conf.n_resize)
    image_stat_processing(conf)
