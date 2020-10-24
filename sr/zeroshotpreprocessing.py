#!/usr/bin/env python3
"""Usage:   zeroshotpreprocessing.py --input-directory=input_path --output-directory=output_path
            --n_resize=no_sample
            zeroshotpreprocessing.py --help | -help | -h

The main usage of zeroshotprepocessing:
 1.) Calculate a kernel for each image from a set of images.
 2.) For each image create a user defined set of samples, where each sample is reduced by a scale factor of 5 %.
 3.) the samples will then be cut into a user defined patch size and saved in LR and HR directories
This will calulate the kernel using KernelGAN and using this kernel will calculate patches and all these data will be
saved in the output directory.

Arguments:
  --input-directory=input_path   : input files
  --output-directory=output_path : output location
  --n_resize=no_sample           : no. of scales generated for the input using a reduction factor of 5%

Options:
  -h --help -h

"""

from cutter import loader, matrix_cutter
from kernelgan import imresize
from kernelgan import train
import numpy as np
import os
from pathlib import Path
import json
from configs import Config
import random

sample_dict = {"--X2": 0.5, "--X4": 0.25, "--X8": 0.125}


def image_stat_processing(conf):
    """

    Parameters
    ----------
    param conf: contains all the configurations required for the preparation

    Returns
    -------

    """
    conf.real_image = True
    output_directory = Path(conf.output_dir_path)
    input_directory = Path(conf.input_dir_path)
    stats = json.load(open(str(input_directory / "stats.json")))

    """
    This loop will read all npz files in a directory
    """
    for image_path in input_directory.rglob("*.npz"):

        """
        this is for creating output folder for each distribution in output directory
        """
        image_name = os.path.splitext(image_path.name)[0]
        image = loader(image_path)



        # reshape the images
        image = image.reshape((image.shape[0], image.shape[1], 1))
        conf.image = image
        conf.stats = stats
        kernel = train(conf)
        sample_list = [image]

        print("The image is being rescaled by given n times")
        for _ in range(conf.n_resize):
            out_image = imresize(im=image, scale_factor=0.95, kernel=kernel)
            sample_list.append(out_image)
            image = out_image


        print("process of cutting and saving images has started")


        # looping over the n samples
        for i, sample in enumerate(sample_list):
            sample = sample[:, :, 0]
            images_cut = matrix_cutter(sample)

            #this is done to create training sets and validation sets for training edsr
            if random.randint(0, 10) > 7:
                data_type = "valid"
            else:
                data_type = "train"
            hr_opath = output_directory / data_type / "HR" / image_name
            lr_opath = output_directory / data_type / "LR" / image_name
            if not os.path.isdir(hr_opath):
                os.makedirs(str(hr_opath))
            if not os.path.isdir(lr_opath):
                os.makedirs(str(lr_opath))
                # saving kernel
                np.save(str(lr_opath / "kernel.npy"), kernel)

                # saving stat file in lr
                with open(str(lr_opath / "stats.json"), "w") as sfile:
                    json.dump(stats, sfile)

                # saving stat file in hr
                with open(str(hr_opath / "stats.json"), "w") as sfile:
                    json.dump(stats, sfile)

            # saving the cut images
            for k, j, mat in images_cut:
                fname = image_name+"_" + str(i) + "_" + str(k) + "_" + str(j)
                np.savez_compressed(lr_opath / fname, mat)
                np.savez_compressed(hr_opath/ fname, mat)
        print("process has finished")

if __name__ == "__main__":
    conf = Config().parse()
    image_stat_processing(conf)
