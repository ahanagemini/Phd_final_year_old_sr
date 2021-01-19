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
import shutil
import numpy as np
import scipy.ndimage
import torch
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
from stat_plotter import PlotStat
from different_loss_plotter import LossPlotter
from train_util import model_selection, check_load_pretrained_model
from vgg_trainer import vgg_process
from vgg_tester import vgg_testing

sample_dict = {"--X2": 0.5, "--X4": 0.25, "--X8": 0.125}

def pretrained_model_upsample(mat, conf):
    """
    This function takes a pretrained model and uses it to upsample
    Parameters
    ----------
    mat
    conf

    Returns
    -------

    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    training_parameters = {}
    height, width = mat.shape
    model = model_selection(conf.pretrained_architecture, conf.aspp, conf.dilation, conf.act)
    training_parameters["pretrained_model"] = model
    training_parameters["pretrained_model_path"] = Path(conf.pretrained_model_path) / "best"
    training_parameters = check_load_pretrained_model(training_parameters)
    model = training_parameters["pretrained_model"]
    mat = np.reshape(mat, (1, 1, height, width))
    mat = torch.from_numpy(mat)
    mat = mat.to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        mat = model(mat)
        mat = mat.cpu().numpy()
        mat = mat[0, 0, :, :]
    return mat

def vgg_hr_random_cropper(image_matrix, hr_opath, fname):
    height_mat, width_mat = image_matrix.shape
    start_height = np.random.randint(0, height_mat - 64)
    end_height = start_height + 64
    start_width = np.random.randint(0, width_mat - 64)
    end_width = start_width + 64
    hr_mat = image_matrix[start_height: end_height, start_width: end_width]
    np.savez_compressed(hr_opath / fname, hr_mat)

def writetext(imgfile):
    img = Image.open(imgfile)
    width, height = img.size
    draw = ImageDraw.Draw(img)
    font_path = os.path.abspath(os.path.expanduser("./font/dancing.ttf"))
    font = ImageFont.truetype(font_path, 10)
    draw.text((0, 0), "KERNEL GAN IMAGE RESIZE", font=font, fill=(0, 255, 0))
    draw.text((width / 2, 0), "PIL IMAGE RESIZE", font=font, fill=(0, 0, 255))
    img.save(imgfile)

def image_clipper(image, stats, factor=0.33):
    """
    This method clips the data max value and min value to 5 % of max value and min value of entire image distribution
    in stats file
    Parameters
    ----------
    image: image matrix
    stats: image statistics

    Returns
    -------

    """
    extreme = max(abs(stats["max"]), abs(stats["min"]))
    max_value = factor * extreme
    min_value = -factor * extreme
    return np.clip(image, min_value, max_value)


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


def saving_different_image_resize_stats(
    save_path, image_name, image, kernel, scale_factor=0.5
):
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

    # creating save folder
    save_path = Path(save_path)
    save_path = save_path / image_name
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    save_path = save_path
    plotter = PlotStat()
    plotter.plot_stat(
        str(save_path), image=image, kernel=kernel, scale_factor=scale_factor
    )


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


def pil_saving_images(sample_list, conf):
    """

    Parameters
    ----------
    sample_list: This contains the list of images that needed to be cut and saved
    conf: configuration
    Returns
    -------

    """
    image_name = conf.image_name
    stats = conf.stats
    output_directory = conf.output_directory
    print("process of cutting and pasting images has started")
    sample_count = len(sample_list)
    resizer = PlotStat()
    # iterating through the sample list
    for i, sample in enumerate(tqdm(sample_list)):
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
            np.savez_compressed(hr_opath / fname, mat)
            if conf.kernel_gan:
                mat = np.reshape(mat, (mat.shape[0], mat.shape[1], 1))
                mat = imresize(
                    im=mat, scale_factor=conf.scale_factor, kernel=conf.kernel
                )
                mat = mat[:, :, 0]
            else:
                mat = resizer.pil_image(mat, conf.scale_factor)
            if conf.vgg:
                mat = pretrained_model_upsample(mat, conf)
            np.savez_compressed(lr_opath / fname, mat)

    print("process of cutting and saving images has ended")


def compare_stat(sample_list, conf):
    """

    Parameters
    ----------
    sample_list
    conf

    Returns
    -------

    """
    print("started stat comparision")
    for i, sample in enumerate(tqdm(sample_list)):
        images_cut = matrix_cutter(sample)
        for k, j, mat in images_cut:
            fname = (
                conf.image_name
                + "_"
                + format(i, "05d")
                + "_"
                + format(k, "05d")
                + "_"
                + format(j, "05d")
            )

            if conf.save_compare_stat:
                save_path = (
                    os.path.dirname(os.path.abspath(__file__)) + r"/comparision_stat"
                )
                saving_different_image_resize_stats(
                    save_path, fname, mat, conf.kernel, scale_factor=conf.scale_factor
                )
    print("ended stat comparision")


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
    image_kernel_tuple = []
    loss_plotter = LossPlotter()
    for kernel_directory, stats, directory_name in kernel_directories:
        kernel_entry = {}
        kernel_save = Path(conf.kernel_save) / directory_name

        for image_path in tqdm(kernel_directory):

            """
            this is for creating output folder for each distribution in output directory
            """
            image_name = os.path.splitext(image_path.name)[0]
            image = loader(image_path)
            image = image_clipper(image, stats)

            # Run kernelgan, compute kernel, then reshape the images
            image = image.reshape((image.shape[0], image.shape[1], 1))
            print("Image shape:", image.shape)
            conf.image = image
            conf.stats = stats
            kernel = check_kernel(conf, directory_name, image_name)
            kernel_entry[image_name] = float(np.sum(image))
            np.save(str(kernel_save / (image_name + ".npy")), kernel)
            sample_list = [image[:, :, 0]]
            print("Image shape:", image.shape)
            print(f"The image is being rescaled by given {conf.n_resize} times")
            scale_factor = 0.95
            for i in range(conf.n_resize):
                scale = scale_factor ** (i + 1)
                out_image = imresize(im=image, scale_factor=scale, kernel=kernel)
                out_image = out_image[:, :, 0]
                sample_list.append(out_image)

                height, width = out_image.shape
                assert height >= 256 and width >= 256

            print("process of cutting and saving images has started")
            np.random.shuffle(sample_list)
            conf.stats = stats
            conf.output_directory = output_directory
            conf.image_name = image_name
            conf.kernel = kernel
            if conf.save_compare_stat:
                compare_stat(sample_list, conf)
            pil_saving_images(sample_list, conf)

            print("the process of creating predict folder has started")
            predict_kernel(image[:, :, 0], image_name, output_directory, stats)
            print("the process of creating predict has ended")

        # saving the kernel entries in json file
        with open(str(kernel_save / "kernel_entry.json"), "w") as kfile:
            json.dump(kernel_entry, kfile)

    print(f"len of image_kernel_tuple = {len(image_kernel_tuple)}")
    save_path = os.path.dirname(os.path.abspath(__file__)) + r"/comparision_stat"
    loss_plotter.ssim_plotter(save_path, image_kernel_tuple, scale_factor=0.25)
    print("kernelgan process has finished")


def perform_pil_image_resize(pil_directories, conf):
    """
    This method will perform pil_image resize on the images
    Parameters
    ----------
    pil_directories: Contains directories that need pil_image_resize
    conf: configuration

    Returns
    -------

    """
    print("pil image resize has begun")
    output_directory = Path(conf.cutting_output_dir_path)
    plotter = PlotStat()
    for pil_directory, stats, directory_name in pil_directories:
        for image_path in tqdm(pil_directory):
            """
            this is for creating output folder for each distribution in output directory
            """
            image_name = os.path.splitext(image_path.name)[0]
            image = loader(image_path)
            image = image_clipper(image, stats)
            scale_factor = 0.95
            sample_list = [image]

            # scaling down 5 % using pil resize
            for i in range(conf.n_resize):
                image = plotter.pil_image(image, scale_factor)
                sample_list.append(image)
                height, width = image.shape[0], image.shape[1]
                assert height >= 256 and width >= 256

            # shuffling the sample list
            np.random.shuffle(sample_list)
            conf.stats = stats
            conf.output_directory = output_directory
            conf.image_name = image_name
            pil_saving_images(sample_list, conf)
            print("the process of creating predict folder has started")
            predict_kernel(image, image_name, output_directory, stats)
            print("the process of creating predict has ended")


def perform_bilinear_and_stats_zoom(scipy_directories, conf):
    """
    Parameters
    ----------
    scipy_directories: Contains the directories that require scipy.ndimage.zoom
    conf: Configuration

    Returns
    -------

    """
    print("scipy zoom process has begun")

    resizer = PlotStat()
    output_directory = Path(conf.cutting_output_dir_path)
    for scipy_directory, stats in scipy_directories:
        scipy_len = len(scipy_directory)
        for i, image_path in enumerate(tqdm(scipy_directory)):
            directory_name = image_path.parent.name
            image_name = os.path.splitext(image_path.name)[0]
            image_matrix = loader(image_path)
            image_matrix = image_clipper(image_matrix, stats)

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
                np.savez_compressed(hr_opath / fname, mat)
                if i < int(0.2*scipy_len):
                    mat = resizer.pil_image(mat, scale_factor=0.25)
                    mat = image_clipper(mat, stats)
                    mat = pretrained_model_upsample(mat, conf)
                    np.savez_compressed(lr_opath / fname, mat)
                elif i > int(0.2 * scipy_len) and i < int(0.4 * scipy_len):
                    mat = resizer.scipy_zoom(mat, scale_factor=0.25)
                    mat = image_clipper(mat, stats)
                    mat = pretrained_model_upsample(mat, conf)
                    np.savez_compressed(lr_opath / fname, mat)
                elif i > int(0.4 * scipy_len) and i < int(0.6 * scipy_len):
                    mat = resizer.t_interpolate(mat, mode="bilinear", scale_factor=0.25)
                    mat = image_clipper(mat, stats)
                    mat = pretrained_model_upsample(mat, conf)
                    np.savez_compressed(lr_opath / fname, mat)
                else:
                    mat = resizer.t_interpolate(mat, mode="bicubic", scale_factor=0.25)
                    mat = image_clipper(mat, stats)
                    mat = pretrained_model_upsample(mat, conf)
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
    print(f"resume file {conf.resume}")
    print(f"cuttingdir {conf.cutting_output_dir_path}")

    if not conf.resume and not conf.load_last_trained:
        # deletting if cutting out existed already to avoid overlaps
        if os.path.isdir(output_directory):
            shutil.rmtree(output_directory)
        # deleting log_dir if starting new experiment
        if os.path.isdir(conf.log_dir):
            shutil.rmtree(conf.log_dir)

        input_directory = Path(conf.input_dir_path)

        # scanning for directories in input directory
        directories = os.scandir(input_directory)

        # gets a dictionary containing directories that need kernelgan and directories that need scipy.ndimage.zoom
        directories_dict = get_kernel_non_kernel_directories(directories)
        if conf.kernel_gan:
            perform_kernelgan(directories_dict["kernel"], conf)
        else:
            perform_pil_image_resize(directories_dict["kernel"], conf)

        perform_bilinear_and_stats_zoom(directories_dict["scipy"], conf)

    train_path = output_directory / "train"
    valid_path = output_directory / "valid"
    test_path = output_directory / "test"
    predict_path = output_directory / "predict"

    if conf.vgg:
        print("started vgg training")
        vgg_process(
            train_path,
            valid_path,
            conf.log_dir,
            conf.architecture,
            conf.num_epochs,
            conf.lognorm,
            conf.aspp,
            conf.dilation,
            conf.act,
            conf.model_save
        )
        print("vgg training has ended")

    else:
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
            conf.model_save
        )
        print("training is complete")

        # EDSR Loading model
        print("started testing")
        best_model_save = Path(conf.model_save)
        print(f"path of trained model is {best_model_save}")
        best_model = sorted(list(best_model_save.rglob("*best_model*")))[-1]

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
