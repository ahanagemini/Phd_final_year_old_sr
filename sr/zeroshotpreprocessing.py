#!/usr/bin/env python3.8
"""Usage:   zeroshotpreprocessing.py --input_dir_path=input_path --output=output_path
            --n_resize=no_sample --scale_factor=scale_factor
            zeroshotpreprocessing.py --help | -help | -h

python3.8 zeroshotpreprocessing.py --input_dir_path=/home/piyush/Dropbox\ \(Intelligent\ Robotics\)/FSU/Research/Sumanth/newsrtest/slices --output=./output --n_resize=10 --kernel_factor='--X4' --num_epochs=100 --architecture="edsr_16_64"

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
  --scale_factor=scale_factor    : no. of

Options:
  -h --help -h

"""
import os
import sys
import json
import random
import shutil
import numpy as np
import scipy.ndimage
from pathlib import Path
from cutter import loader, matrix_cutter
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from kernelgan import imresize
from kernelgan import train as kernelgan_train
from configs import Config
from trainer import process
from tester import evaluate
from tqdm import tqdm

sample_dict = {"--X2": 0.5, "--X4": 0.25, "--X8": 0.125}


def writetext(imgfile):
    img = Image.open(imgfile)
    width, height = img.size
    draw = ImageDraw.Draw(img)
    font_path = os.path.abspath(os.path.expanduser("./font/dancing.ttf"))
    font = ImageFont.truetype(font_path, 10)
    draw.text((0, 0), "KERNEL GAN IMAGE RESIZE", font=font, fill=(0, 255, 0))
    draw.text((width / 2, 0), "PIL IMAGE RESIZE", font=font, fill=(0, 0, 255))
    img.save(imgfile)


def stat_calculator(input_path):
    print("creating stats")
    total_sum = 0.0
    total_square_sum = 0.0
    total_count = 0.0
    image_min = 0
    image_max = 0
    for image_path in input_path.rglob("*.npz"):
        imatrix = loader(image_path)
        matrix_vector = np.asarray(imatrix).reshape(-1)
        square_vector = np.square(matrix_vector, dtype=np.float)
        matrix_sum = np.sum(matrix_vector, dtype=np.float)
        square_sum = np.sum(square_vector, dtype=np.float)
        matrix_count = len(matrix_vector)

        # this information is for total mean calculation
        total_sum = total_sum + matrix_sum
        total_square_sum = total_square_sum + square_sum
        total_count = total_count + matrix_count

        # maximum and minimum
        matrix_max = np.max(matrix_vector)
        matrix_min = np.min(matrix_vector)
        if image_max < matrix_max:
            image_max = matrix_max

        if image_min > matrix_min:
            image_min = matrix_min
    if total_count == 0:
        print("no elements loaded creating stats failed")
        return
    total_mean = total_sum / total_count
    total_variance = (total_square_sum / total_count) - (total_sum / total_count) ** 2
    stats = {}
    stats["mean"] = total_mean
    stats["std"] = np.sqrt(total_variance)
    stats["max"] = float(image_max)
    stats["min"] = float(image_min)

    return stats


def assert_stats(input_directory):
    """
    Returns stats. If stats.json is not present, computes it.
    """
    # input_directory = Path(conf.input_dir_path)
    if not os.path.isfile(str(input_directory / "stats.json")):
        """ calculate stats"""
        stats = stat_calculator(input_directory)

        with open(str(input_directory / "stats.json"), "w") as sfile:
            json.dump(stats, sfile)

    else:
        print("loading available stats")
        stats = json.load(open(str(input_directory / "stats.json")))
    return stats


def get_kernel_non_kernel_directories(directories):
    """

    Parameters
    ----------
    directories: it contains all the folders present in input directory

    Returns
    -------
    directories_dict
    """

    directories_dict = {}

    # directories on which kernel gan will be applied
    directories_dict["kernel"] = []

    # directories on which scipy.ndimage.zoom will be applied
    directories_dict["scipy"] = []

    for directory in directories:
        directory = Path(directory)
        temp_directory = list(directory.rglob("*.npz"))
        if len(temp_directory) <= 15:
            stats = assert_stats(directory)
            directory_name = directory.name
            directories_dict["kernel"].append((temp_directory, stats, directory_name))
        else:
            stats = assert_stats(directory)
            directories_dict["scipy"].append((temp_directory, stats))
    return directories_dict


def predict_kernel(image_matrix, image_name, output_directory, stats):
    """
    This function cuts the original image to size 64, 64 and stores the HR as dummy 256, 256 and LR as 64, 64

    Parameters
    ----------
    image_matrix: The original image matrix
    image_name: Name of the Image
    output_directory: the directory where the data needs to be stored

    Returns
    -------
    """

    image_cuts = matrix_cutter(image_matrix, width=64, height=64)

    data_type = "predict"
    hr_opath = output_directory / data_type / "HR" / image_name
    lr_opath = output_directory / data_type / "LR" / image_name
    if not os.path.isdir(hr_opath):
        os.makedirs(str(hr_opath))
    if not os.path.isdir(lr_opath):
        os.makedirs(str(lr_opath))

    with open(str(lr_opath / "stats.json"), "w") as sfile:
        json.dump(stats, sfile)

    with open(str(hr_opath / "stats.json"), "w") as sfile:
        json.dump(stats, sfile)

    for g, j, imat in image_cuts:
        fname = image_name + "_" + format(g, "05d") + "_" + format(j, "05d")
        np.savez_compressed(lr_opath / fname, imat)
        imat = scipy.ndimage.zoom(imat, 4)
        np.savez_compressed(hr_opath / fname, imat)


def check_kernel(conf, directory_name, image_name):
    """
    This method will check if kernel gan was run before and if it was it will directly load the saved kernel and if not
    it will run kernelgan. Finally it will return kernel.

    Parameters
    ----------
    conf : contains the configurations required by kernelgan
    directory_name : contains the name of the directory
    image_name: contains the name of the image

    Returns
    -------
    kernel
    """

    kernel_save = Path(conf.kernel_save) / directory_name

    # checking if the directory exists
    if not os.path.isdir(kernel_save):
        os.makedirs(kernel_save)
        kernel = kernelgan_train(conf)
        return kernel
    else:
        # checking if kernel entry file exists
        if not os.path.isfile(kernel_save / "kernel_entry.json"):
            kernel = kernelgan_train(conf)
            return kernel
        else:
            images_data = json.load(open(str(kernel_save / "kernel_entry.json")))

            # checking if the image details is present in the entry
            if image_name in images_data.keys():
                data_image_sum = images_data[image_name]
                actual_image_sum = float(np.sum(conf.image))

                # checking if the kernel belongs to the correct image
                if actual_image_sum == data_image_sum:
                    kernel = np.load(str(kernel_save / (image_name + ".npy")))
                    return kernel
                else:
                    kernel = kernelgan_train(conf)
                    return kernel
            else:
                kernel = kernelgan_train(conf)
                return kernel

def saving_different_image_resize_stats(save_path, image_name, image, kernel):
    """

    Parameters
    ----------
    save_path
    image_name
    image
    kernel

    Returns
    -------
    """

    stat_image = {}
    kernel_image = image
    scipy_image = image
    pil_image = Image.fromarray(image)

    # creating save folder
    save_path = Path(save_path)
    save_path = save_path / image_name
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # kernel resize
    kernel_image = kernel_image.reshape((kernel_image.shape[0], kernel_image.shape[1], 1))
    kernel_image = imresize(im=kernel_image, scale_factor=0.5, kernel=kernel)
    kernel_image = kernel_image[:, :, 0]

    # PIL resize
    pil_image = pil_image.resize((128, 128))
    pil_image = np.array(pil_image)

    # scipy ndzoom
    scipy_image = scipy.ndimage.zoom(scipy_image, 0.5)

    # mean calculation
    stat_image[image_name+"_mean"] = float(np.mean(image))
    stat_image[image_name+"_kernel_image_mean"] = float(np.mean(kernel_image))
    stat_image[image_name+"_pil_image_resize_mean"] = float(np.mean(pil_image))
    stat_image[image_name+"_scipy_image_mean"] = float(np.mean(scipy_image))
    # storing in list for plot
    mean_list = [stat_image[image_name+"_mean"], stat_image[image_name+"_kernel_image_mean"],
                 stat_image[image_name+"_pil_image_resize_mean"], stat_image[image_name+"_scipy_image_mean"]]

    # std calculation
    stat_image[image_name+"_std"] = float(np.std(image))
    stat_image[image_name+"_kernel_image_std"] = float(np.std(kernel_image))
    stat_image[image_name+"_pil_image_resize_std"] = float(np.std(pil_image))
    stat_image[image_name+"_scipy_image_std"] = float(np.std(scipy_image))
    # storing in list for plot
    std_list = [stat_image[image_name+"_std"], stat_image[image_name+"_kernel_image_std"],
                stat_image[image_name+"_pil_image_resize_std"], stat_image[image_name+"_scipy_image_std"]]

    # max calculation
    stat_image[image_name+"_max"] = float(np.max(image))
    stat_image[image_name+"_kernel_image_max"] = float(np.max(kernel_image))
    stat_image[image_name+"_pil_image_max"] = float(np.max(pil_image))
    stat_image[image_name+"_scipy_image_max"] = float(np.max(scipy_image))
    # storing in list for plot
    max_list = [stat_image[image_name+"_max"], stat_image[image_name+"_kernel_image_max"],
                stat_image[image_name+"_pil_image_max"], stat_image[image_name+"_scipy_image_max"]]

    # min calculation
    stat_image[image_name+"_min"] = float(np.min(image))
    stat_image[image_name+"_kernel_image_min"] = float(np.min(kernel_image))
    stat_image[image_name+"_pil_image_min"] = float(np.min(pil_image))
    stat_image[image_name+"_scipy_image_min"] = float(np.min(scipy_image))
    # storing in list for plot
    min_list = [stat_image[image_name+"_min"], stat_image[image_name+"_kernel_image_min"],
                stat_image[image_name+"_pil_image_min"], stat_image[image_name+"_scipy_image_min"]]

    fig_mean = plt.figure()
    plt.title("Mean Plot for different type of images")
    plt.ylabel("mean")
    plt.xlabel("different types of images")
    x_values = ["ground_truth", "kernel_image", "PIL_image", "Scipy_image"]
    plt.bar(x_values, mean_list, width = 0.4)
    fig_mean.savefig(str(save_path / ("mean_plot.png")))
    plt.close(fig_mean)

    fig_std = plt.figure()
    plt.title("std plot for different types of images")
    plt.ylabel("standard deviation")
    plt.xlabel("different types of images")
    x_values = ["ground_truth", "kernel_image", "PIL_image", "Scipy_image"]
    plt.bar(x_values, std_list, width=0.4)
    fig_std.savefig(str(save_path / ("std_plot.png")))
    plt.close(fig_std)

    fig_max = plt.figure()
    plt.title("max plot for different types of images")
    plt.ylabel("max pixel")
    plt.xlabel("different type of images")
    x_values = ["ground_truth", "kernel_image", "PIL_image", "Scipy_image"]
    plt.bar(x_values, max_list, width=0.4)
    fig_max.savefig(str(save_path / ("max_plot.png")))
    plt.close(fig_max)

    fig_min = plt.figure()
    plt.title("min plot for different images")
    plt.ylabel("min pixel")
    plt.ylabel("different types of images")
    x_values = ["ground_truth", "kernel_image", "PIL_image", "Scipy_image"]
    plt.bar(x_values, min_list, width=0.4)
    fig_min.savefig(str(save_path / ("min_plot.png")))
    plt.close(fig_min)

    save_path_stats = save_path / "stats.json"
    with open(str(save_path_stats), "w") as sfile:
        json.dump(stat_image, sfile)

    del [stat_image, kernel_image, pil_image, scipy_image, fig_max, fig_mean, fig_std, fig_min]




def compare_images(save_path, image_name, image_1, image_2, stat):
    """

    Parameters
    ----------
    save_path
    image_1
    image_2
    stat

    Returns
    -------

    """
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    save_path = save_path + f"/{image_name}.png"
    save_plots = np.hstack([image_1, image_2])
    save_plots = np.clip(save_plots, stat["min"], stat["max"])
    vmax = stat["mean"] + 3 * stat["std"]
    vmin = stat["min"]
    plt.imsave(save_path, save_plots, vmin=vmin, vmax=vmax, cmap="gray")
    writetext(imgfile=save_path)


def perform_kernelgan(kernel_directories, conf):
    """

    Parameters
    ----------
    kernel_directories: Contains directories that need kernelgan
    conf: Configuration

    Returns
    -------

    """

    print("kernelgan process has started")
    output_directory = Path(conf.cutting_output_dir_path)
    for kernel_directory, stats, directory_name in kernel_directories:

        kernel_entry = {}
        kernel_save = Path(conf.kernel_save) / directory_name
        for image_path in tqdm(kernel_directory):

            """
            this is for creating output folder for each distribution in output directory
            """
            image_name = os.path.splitext(image_path.name)[0]
            image = loader(image_path)

            # Run kernelgan, compute kernel, then reshape the images
            image = image.reshape((image.shape[0], image.shape[1], 1))
            print("Image shape:", image.shape)
            conf.image = image
            conf.stats = stats
            kernel = check_kernel(conf, directory_name, image_name)
            kernel_entry[image_name] = float(np.sum(image))
            np.save(str(kernel_save / (image_name + ".npy")), kernel)
            sample_list = [image]
            print("Image shape:", image.shape)
            print(f"The image is being rescaled by given {conf.n_resize} times")
            scale_factor = 0.95
            for i in range(conf.n_resize):
                scale = scale_factor ** (i + 1)
                out_image = imresize(im=image, scale_factor=scale, kernel=kernel)
                sample_list.append(out_image)

                height, width = out_image.shape[0], out_image.shape[1]
                assert height >= 256 and width >= 256

            print("process of cutting and saving images has started")
            sample_count = len(sample_list)
            np.random.shuffle(sample_list)
            # looping over the n samples
            for i, sample in enumerate(tqdm(sample_list)):
                sample = sample[:, :, 0]
                images_cut = matrix_cutter(sample)

                # this is done to create training sets and validation sets for training edsr
                if i > int(0.7 * sample_count) and i < int(0.9 * sample_count):
                    data_type = "valid"
                elif i >= int(0.9 * sample_count):
                    data_type = "test"
                else:
                    data_type = "train"
                hr_opath = output_directory / data_type / "HR" / image_name
                lr_opath = output_directory / data_type / "LR" / image_name
                if not os.path.isdir(hr_opath):
                    os.makedirs(str(hr_opath))
                if not os.path.isdir(lr_opath):
                    os.makedirs(str(lr_opath))

                # saving stat file in lr
                with open(str(lr_opath / "stats.json"), "w") as sfile:
                    json.dump(stats, sfile)

                # saving stat file in hr
                with open(str(hr_opath / "stats.json"), "w") as sfile:
                    json.dump(stats, sfile)

                # saving the cut images
                for k, j, mat in images_cut:
                    fname = (
                        image_name
                        + "_"
                        + format(i, "05d")
                        + "_"
                        + format(k, "05d")
                        + "_"
                        + format(j, "05d")
                    )
                    if conf.compare:
                        img_mat = mat
                        img = Image.fromarray(img_mat)
                        img = img.resize((128, 128))
                        img_mat = np.reshape(mat, (mat.shape[0], mat.shape[1], 1))

                        img_mat = imresize(im=img_mat, scale_factor=0.5, kernel=kernel)
                        img_mat = img_mat[:, :, 0]
                        save_path = os.path.dirname(__file__) + r"/comparision"
                        compare_images(save_path, fname, img_mat, img, stats)

                    if conf.save_compare_stat:
                        save_path = os.path.dirname(os.path.abspath(__file__)) + r"/comparision_stat"
                        saving_different_image_resize_stats(save_path, fname, mat, kernel)

                    np.savez_compressed(hr_opath / fname, mat)
                    mat = np.reshape(mat, (mat.shape[0], mat.shape[1], 1))

                    mat = imresize(
                        im=mat, scale_factor=conf.scale_factor, kernel=kernel
                    )
                    mat = mat[:, :, 0]
                    np.savez_compressed(lr_opath / fname, mat)

            print("the process of creating predict folder has started")
            predict_kernel(image[:, :, 0], image_name, output_directory, stats)
            print("the process of creating predict has ended")

        # saving the kernel entries in json file
        with open(str(kernel_save / "kernel_entry.json"), "w") as kfile:
            json.dump(kernel_entry, kfile)

    print("kernelgan process has finished")


def perform_scipy_ndimage_zoom(scipy_directories, conf):
    """

    Parameters
    ----------
    scipy_directories: Contains the directories that require scipy.ndimage.zoom
    conf: Configuration

    Returns
    -------

    """
    print("scipy zoom process has begun")

    output_directory = Path(conf.cutting_output_dir_path)
    for scipy_directory, stats in scipy_directories:
        scipy_len = len(scipy_directory)
        for i, image_path in enumerate(tqdm(scipy_directory)):
            directory_name = image_path.parent.name
            image_name = os.path.splitext(image_path.name)[0]
            image_matrix = loader(image_path)
            images_cut = matrix_cutter(image_matrix)

            if i < int(0.9 * scipy_len):
                data_type = "train"
            elif i >= int(0.9 * scipy_len) and i < int(0.95 * scipy_len):
                data_type = "valid"
            else:
                data_type = "test"

            hr_opath = output_directory / data_type / "HR" / directory_name
            lr_opath = output_directory / data_type / "LR" / directory_name
            if not os.path.isdir(hr_opath):
                os.makedirs(str(hr_opath))
            if not os.path.isdir(lr_opath):
                os.makedirs(str(lr_opath))

            # saving stat file in lr
            with open(str(lr_opath / "stats.json"), "w") as sfile:
                json.dump(stats, sfile)

            # saving stat file in hr
            with open(str(hr_opath / "stats.json"), "w") as sfile:
                json.dump(stats, sfile)

            for k, j, mat in images_cut:
                fname = (
                    image_name
                    + "_"
                    + format(i, "05d")
                    + "_"
                    + format(k, "05d")
                    + "_"
                    + format(j, "05d")
                )
                if np.random.randint(1, 100) < 50:
                    img = Image.fromarray(mat)
                    img = img.resize((64, 64))
                    np.savez_compressed(hr_opath / fname, np.array(img))
                else:
                    np.savez_compressed(hr_opath / fname, mat)
                    mat = scipy.ndimage.zoom(mat, 0.25)
                    np.savez_compressed(lr_opath / fname, mat)

    print("scipy zoom process has finished")


def image_stat_processing(conf):
    """

    Parameters
    ----------
    param conf: contains all the configurations required for the preparation

    Returns
    -------

    """
    conf.real_image = True
    output_directory = Path(conf.cutting_output_dir_path)

    # deletting if cutting out existed already to avoid overlaps
    if os.path.isdir(output_directory):
        shutil.rmtree(output_directory)
    input_directory = Path(conf.input_dir_path)

    # scanning for directories in input directory
    directories = os.scandir(input_directory)

    # gets a dictionary containing directories that need kernelgan and directories that need scipy.ndimage.zoom
    directories_dict = get_kernel_non_kernel_directories(directories)

    perform_kernelgan(directories_dict["kernel"], conf)
    perform_scipy_ndimage_zoom(directories_dict["scipy"], conf)

    train_path = output_directory / "train"
    valid_path = output_directory / "valid"
    test_path = output_directory / "test"
    predict_path = output_directory / "predict"

    # EDSR Training
    print("started EDSR Training")
    process(
        train_path,
        valid_path,
        conf.log_dir,
        conf.architecture,
        conf.num_epochs,
        conf.lognorm,
        conf.debug_pics,
        conf.aspp,
        conf.dilation,
        conf.act,
        conf.model_save,
        kernel=True,
    )
    print("training is complete")

    # EDSR Loading model
    print("started testing")
    best_model_save = Path(conf.model_save)
    best_model = sorted(list(best_model_save.rglob("*best_model.pt")))[-1]

    final_test_path = Path(conf.output_dir_path) / "Result_Test"
    if not os.path.isdir(final_test_path):
        os.makedirs(final_test_path)
    args = {
        "--input": test_path,
        "--output": final_test_path,
        "--architecture": conf.architecture,
        "--model": best_model,
        "--act": conf.act,
        "--lognorm": conf.lognorm,
        "--active": conf.active,
        "--save_slice": conf.save_slice,
        "--aspp": conf.aspp,
        "--dilation": conf.dilation,
        "kernel": True,
        "hr": True,
    }
    evaluate(args)
    print("finished testing exiting")

    print("started predict testing")
    final_predict_path = Path(conf.output_dir_path) / "Result_Predict"
    if not os.path.isdir(final_predict_path):
        os.makedirs(final_predict_path)
    args = {
        "--input": predict_path,
        "--output": final_predict_path,
        "--architecture": conf.architecture,
        "--model": best_model,
        "--act": conf.act,
        "--lognorm": conf.lognorm,
        "--active": conf.active,
        "--save_slice": conf.save_slice,
        "--aspp": conf.aspp,
        "--dilation": conf.dilation,
        "kernel": True,
        "hr": True,
    }
    evaluate(args)
    print("finished testing exiting")


if __name__ == "__main__":
    if (
        len(sys.argv) == 1
        or "--input_dir_path" not in str(sys.argv)
        or "--output" not in str(sys.argv)
    ):
        sys.argv.append("-h")
    conf = Config().parse()
    print(f"{str(conf.model_save)}")
    image_stat_processing(conf)
