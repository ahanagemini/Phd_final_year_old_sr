import torch
import torch.nn
import torch.nn.functional as F
from scipy.ndimage import zoom
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import os
from kernelgan import imresize
from skimage import metrics


class PlotStat:
    """This class plots the graphs for stats"""

    def t_interpolate(self, image, mode, scale_factor):
        """

        Parameters
        ----------
        image
        mode

        Returns
        -------

        """
        image = image.reshape((image.shape[0], image.shape[1], 1))
        image = image.reshape((1, 1, image.shape[0], image.shape[1]))
        image = torch.tensor(image, dtype=torch.float32)
        if mode == "nearest":
            image = F.interpolate(
                image,
                scale_factor=scale_factor,
                mode=mode,
                recompute_scale_factor=False,
            )
        else:
            image = F.interpolate(
                image,
                scale_factor=scale_factor,
                mode=mode,
                align_corners=True,
                recompute_scale_factor=False,
            )

        image = image.numpy()
        image = image[0, 0, :, :]
        return image

    def pil_image(self, image, scale_factor):
        """

        Parameters
        ----------
        image
        image_shape

        Returns
        -------

        """
        height, width = image.shape
        height, width = int(scale_factor * height), int(scale_factor * width)
        image = Image.fromarray(image)
        image = image.resize((height, width))
        image = np.array(image)
        return image

    def scipy_zoom(self, image, scale_factor):
        """

        Parameters
        ----------
        image
        scale_factor

        Returns
        -------

        """
        image = zoom(image, scale_factor)
        return image

    def kernel_gan_resize(self, image, kernel, scale_factor):
        """

        Parameters
        ----------
        image
        kernel
        scale_factor

        Returns
        -------

        """
        image = image.reshape((image.shape[0], image.shape[1], 1))
        image = imresize(image, scale_factor=scale_factor, kernel=kernel)
        image = image[:, :, 0]
        return image

    def plot_stat(self, plot_save_path, image, kernel, scale_factor):
        """

        Parameters
        ----------
        plot_save_path : directory at which you want to save plots
        image : ground truth image
        kernel : image kernel calculated from kernelgan
        scale_factor : the amount of reduction required in image (between 0 to 1)

        Returns
        -------

        """
        height, width = image.shape
        list_of_plots = [
            ("ground_thruth", image),
            ("pil_image_resize", self.pil_image(image, scale_factor)),
            ("scipy_image_resize", self.scipy_zoom(image, scale_factor)),
            (
                "kernel_gan_resize",
                self.kernel_gan_resize(image, kernel=kernel, scale_factor=scale_factor),
            ),
            ("bicubic", self.t_interpolate(image, "bicubic", scale_factor)),
            ("bilinear", self.t_interpolate(image, "bilinear", scale_factor)),
            ("nearest", self.t_interpolate(image, "nearest", scale_factor)),
        ]

        self.box_plot(plot_save_path + f"/boxplot.png", list_of_plots)
        self.mean_bar_plot(plot_save_path + f"/mean_plot.png", list_of_plots)
        self.std_bar_plot(plot_save_path + f"/std_bar_plot.png", list_of_plots)
        list_of_plots = [
            ("ground_thruth", image),
            (
                "pil_image_resize",
                self.pil_image(self.pil_image(image, scale_factor), 1 // scale_factor),
            ),
            (
                "scipy_image_resize",
                self.scipy_zoom(
                    self.scipy_zoom(image, scale_factor), 1 // scale_factor
                ),
            ),
            (
                "kernel_gan_resize",
                self.kernel_gan_resize(image, kernel=kernel, scale_factor=1),
            ),
            (
                "bicubic",
                self.t_interpolate(
                    self.t_interpolate(image, "bicubic", scale_factor),
                    "bicubic",
                    1 // scale_factor,
                ),
            ),
            (
                "bilinear",
                self.t_interpolate(
                    self.t_interpolate(image, "bilinear", scale_factor),
                    "bilinear",
                    1 // scale_factor,
                ),
            ),
            (
                "nearest",
                self.t_interpolate(
                    self.t_interpolate(image, "nearest", scale_factor),
                    "nearest",
                    1 // scale_factor,
                ),
            ),
        ]

        self.image_save(plot_save_path, list_of_plots, image)

    def mean_bar_plot(self, save_path, plot_list):
        """

        Parameters
        ----------
        plot_list

        Returns
        -------

        """
        label_to_mean_map = {}
        for i, plotter in enumerate(plot_list):
            x_label, image = plotter
            image = np.ravel(image)
            image_mean = np.mean(image)
            label_to_mean_map[x_label] = image_mean

        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111)
        ax.bar(label_to_mean_map.keys(), label_to_mean_map.values())
        fig.savefig(save_path)
        plt.close(fig)

    def std_bar_plot(self, save_path, plot_list):
        """

        Parameters
        ----------
        save_path
        plot_list

        Returns
        -------

        """
        label_to_std_map = {}
        for i, plotter in enumerate(plot_list):
            x_label, image = plotter
            image = np.ravel(image)
            image_std = np.std(image)
            label_to_std_map[x_label] = image_std

        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111)
        ax.bar(label_to_std_map.keys(), label_to_std_map.values())
        fig.savefig(save_path)
        plt.close(fig)

    def box_plot(self, save_path, plot_list):
        """

        Parameters
        ----------
        save_path
        plot_list

        Returns
        -------

        """
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111)
        for i, plotter in enumerate(plot_list):
            x_label, image = plotter
            image = np.ravel(image)
            box_prop = dict(
                color=(np.random.random(), np.random.random(), np.random.random())
            )
            flier_prop = dict(
                markeredgecolor=(
                    np.random.random(),
                    np.random.random(),
                    np.random.random(),
                )
            )
            whisker_prop = dict(
                color=(np.random.random(), np.random.random(), np.random.random())
            )

            median_prop = dict(
                color=(np.random.random(), np.random.random(), np.random.random())
            )

            cap_prop = dict(
                color=(np.random.random(), np.random.random(), np.random.random())
            )

            ax.boxplot(
                image,
                positions=[i],
                labels=[x_label],
                boxprops=box_prop,
                medianprops=median_prop,
                whiskerprops=whisker_prop,
                capprops=cap_prop,
                flierprops=flier_prop,
            )

        fig.savefig(save_path)
        plt.close(fig=fig)

    def image_save(self, save_path, plot_list, ground_truth_image):
        """

        Parameters
        ----------
        save_path
        plot_list
        ground_truth_image

        Returns
        -------

        """
        for i, plotter in enumerate(plot_list):
            image_label, image = plotter
            vmax = np.max(image)
            vmin = np.min(image)
            imgfile = save_path + f"/{image_label}.png"
            plt.imsave(imgfile, image, vmax=vmax, vmin=vmin, cmap="gray")
            with np.errstate(divide="ignore"):
                l1_loss = np.mean(np.abs(ground_truth_image - image))
                psnr_loss = metrics.peak_signal_noise_ratio(
                    ground_truth_image, image, data_range=vmax
                )
                ssim_loss = metrics.structural_similarity(
                    ground_truth_image, image, data_range=vmax
                )
            self.writetext(imgfile, l1_loss, psnr_loss, ssim_loss)

    def writetext(self, imgfile, l1, psnr, ssim):
        img = Image.open(imgfile)
        draw = ImageDraw.Draw(img)
        font_path = os.path.abspath(os.path.expanduser("./font/dancing.ttf"))
        font = ImageFont.truetype(font_path, 10)
        draw.text((0, 0), "L1=" + str(l1), font=font, fill=(255, 0, 0))
        draw.text((0, 16), "PSNR=" + str(psnr), font=font, fill=(255, 0, 0))
        draw.text((0, 32), "SSIM=" + str(ssim), font=font, fill=(255, 0, 0))
        img.save(imgfile)
