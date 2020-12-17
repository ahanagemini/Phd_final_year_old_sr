import torch
import torch.nn
import torch.nn.functional as F
from scipy.ndimage import zoom
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import metrics
from functools import partial
from stat_plotter import PlotStat
from tqdm import tqdm


class LossPlotter:
    """ This class will contain functions to plot"""

    def ssim_plotter(self, save_path, ground_truth_images, scale_factor=0.5):
        """
        This function will plot the ssim

        Parameters
        ----------
        ground_truth_images: list that contains all the ground truth images along with its tuple

        Returns
        -------

        """
        # creating ssim directory
        ssim_save_path = save_path + r"/ssim"
        if not os.path.isdir(ssim_save_path):
            os.makedirs(ssim_save_path)

        # creating psnr directory
        psnr_save_path = save_path + r"/psnr"
        if not os.path.isdir(psnr_save_path):
            os.makedirs(psnr_save_path)

        # creating l1 directory
        l1_loss_save_path = save_path + r"/l1_loss"
        if not os.path.isdir(l1_loss_save_path):
            os.makedirs(l1_loss_save_path)

        interpol = PlotStat()

        # interpolation list
        interpolation_list = [
            "bicubic",
            "bilinear",
            "pil_image_resize",
            "kernel_gan",
            "nearest",
            "scipy_nd_zoom",
        ]

        # creating an empty list for each interpolation category
        interpolation_ssim = {}
        interpolation_l1_loss = {}
        interpolation_psnr_loss = {}
        for interpolation_type in interpolation_list:
            interpolation_ssim[interpolation_type] = []
            interpolation_psnr_loss[interpolation_type] = []
            interpolation_l1_loss[interpolation_type] = []

        # creating partial functions
        interpolation_tuple = [
            (
                "bicubic",
                partial(
                    interpol.t_interpolate, mode="bicubic", scale_factor=scale_factor
                ),
                partial(
                    interpol.t_interpolate,
                    mode="bicubic",
                    scale_factor=1 // scale_factor,
                ),
            ),
            (
                "bilinear",
                partial(
                    interpol.t_interpolate, mode="bilinear", scale_factor=scale_factor
                ),
                partial(
                    interpol.t_interpolate,
                    mode="bilinear",
                    scale_factor=1 // scale_factor,
                ),
            ),
            (
                "pil_image_resize",
                partial(interpol.pil_image, scale_factor=scale_factor),
                partial(interpol.pil_image, scale_factor=1 // scale_factor),
            ),
            (
                "nearest",
                partial(
                    interpol.t_interpolate, mode="nearest", scale_factor=scale_factor
                ),
                partial(
                    interpol.t_interpolate,
                    mode="nearest",
                    scale_factor=1 // scale_factor,
                ),
            ),
            (
                "scipy_nd_zoom",
                partial(interpol.scipy_zoom, scale_factor=scale_factor),
                partial(interpol.scipy_zoom, scale_factor=1 // scale_factor),
            ),
        ]

        # storing the ssim loss for each interpolation
        print("started losses calculation")
        for image_tup in tqdm(ground_truth_images):
            ground_image, kernel = image_tup
            for interpolation_tup in interpolation_tuple:
                label, func_down, func_up = interpolation_tup
                image = func_down(ground_image)
                image = func_up(image)
                vmax = np.max(ground_image)

                # loss calculations
                ssim_loss = metrics.structural_similarity(
                    ground_image, image, data_range=vmax
                )
                psnr_loss = metrics.peak_signal_noise_ratio(
                    ground_image, image, data_range=vmax
                )
                l1_loss = np.mean(np.abs(ground_image - image))

                interpolation_ssim[label].append(ssim_loss)
                interpolation_psnr_loss[label].append(psnr_loss)
                interpolation_l1_loss[label].append(l1_loss)

            # separate section for kernel gan
            kernel_image = interpol.kernel_gan_resize(ground_image, kernel, 1)
            vmax = np.max(ground_image)
            ssim_loss = metrics.structural_similarity(
                ground_image, kernel_image, data_range=vmax
            )
            psnr_loss = metrics.peak_signal_noise_ratio(
                ground_image, kernel_image, data_range=vmax
            )
            l1_loss = np.mean(np.abs(ground_image - kernel_image))
            interpolation_ssim["kernel_gan"].append(ssim_loss)
            interpolation_psnr_loss["kernel_gan"].append(psnr_loss)
            interpolation_l1_loss["kernel_gan"].append(l1_loss)
        print("ended losses calculation")

        print("started ssim plotting")
        # plotting the ssim lists in graph
        fig = plt.figure(figsize=(15, 15))
        plt.title("ssim of different interpolations")
        ax_ssim = fig.add_subplot(111)
        for label, loss in interpolation_ssim.items():
            loss = sorted(loss)
            x = range(len(loss))
            ax_ssim.plot(x, loss, label=label)
        ax_ssim.legend()
        fig.savefig(ssim_save_path + r"/ssim_plot.png")
        plt.close(fig=fig)
        print("ended ssim plotting")

        print("started psnr plotting")
        # plotting the ssim lists in graph
        fig = plt.figure(figsize=(15, 15))
        plt.title("psnr of different interpolations")
        ax_psnr = fig.add_subplot(111)
        for label, loss in interpolation_psnr_loss.items():
            loss = sorted(loss)
            x = range(len(loss))
            ax_psnr.plot(x, loss, label=label)
        ax_psnr.legend()
        fig.savefig(psnr_save_path + r"/psnr_plot.png")
        plt.close(fig=fig)

        print("started l1_loss plotting")
        # plotting the ssim lists in graph
        fig = plt.figure(figsize=(15, 15))
        plt.title("l1_loss of different interpolations")
        ax_l1 = fig.add_subplot(111)
        for label, loss in interpolation_l1_loss.items():
            loss = sorted(loss)
            x = range(len(loss))
            ax_l1.plot(x, loss, label=label)
        ax_l1.legend()
        fig.savefig(l1_loss_save_path + r"/l1_loss_plot.png")
        plt.close(fig=fig)
