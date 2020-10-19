import json

import torch
from .loss import (
    GANLoss,
    DownScaleLoss,
    SumOfWeightsLoss,
    BoundariesLoss,
    CentralizedLoss,
    SparsityLoss,
)
from .networks import Generator, Discriminator, weights_init_G, weights_init_D
import torch.nn.functional as F
from torchsummary import summary
from .util import save_final_kernel, post_process_k, read_image
from scipy.ndimage import filters
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageFont, ImageDraw


class KernelGAN:
    # Constraint co-efficients
    lambda_sum2one = 0.5
    lambda_bicubic = 5
    lambda_boundaries = 0.5
    lambda_centralized = 0
    lambda_sparse = 0

    def __init__(self, conf):
        # Acquire configuration
        self.conf = conf

        # Define the GAN
        self.G = Generator(conf).cuda()
        summary(self.G, (1, 256, 256), batch_size=1, device="cuda")
        self.D = Discriminator(conf).cuda()
        summary(self.D, (1, 256, 256), batch_size=1, device="cuda")

        # Calculate D's input & output shape according to the shaving done by the networks
        self.X4 = conf.X4
        self.d_input_shape = self.G.output_size
        self.d_output_shape = self.d_input_shape - self.D.forward_shave

        # Input tensors
        self.g_input = torch.FloatTensor(
            1, 1, conf.input_crop_size, conf.input_crop_size
        ).cuda()
        self.d_input = torch.FloatTensor(
            1, 1, self.d_input_shape, self.d_input_shape
        ).cuda()

        # The kernel G is imitating
        self.curr_k = torch.FloatTensor(conf.G_kernel_size, conf.G_kernel_size).cuda()

        # Losses
        self.GAN_loss_layer = GANLoss(d_last_layer_size=self.d_output_shape).cuda()
        self.bicubic_loss = DownScaleLoss(scale_factor=conf.scale_factor).cuda()
        self.sum2one_loss = SumOfWeightsLoss().cuda()
        self.boundaries_loss = BoundariesLoss(k_size=conf.G_kernel_size).cuda()
        self.centralized_loss = CentralizedLoss(
            k_size=conf.G_kernel_size, scale_factor=conf.scale_factor
        ).cuda()
        self.sparse_loss = SparsityLoss().cuda()
        self.loss_bicubic = 0

        # Define loss function
        self.criterionGAN = self.GAN_loss_layer.forward

        # Initialize networks weights
        self.G.apply(weights_init_G)
        self.D.apply(weights_init_D)

        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            self.G.parameters(), lr=conf.g_lr, betas=(conf.beta1, 0.999)
        )
        self.optimizer_D = torch.optim.Adam(
            self.D.parameters(), lr=conf.d_lr, betas=(conf.beta1, 0.999)
        )

        print("*" * 60 + '\nSTARTED KernelGAN on: "%s"...' % conf.input_image_path)

    # noinspection PyUnboundLocalVariable
    def calc_curr_k(self):
        """given a generator network, the function calculates the kernel it is imitating"""
        delta = torch.Tensor([1.0]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
        for ind, w in enumerate(self.G.parameters()):
            curr_k = (
                F.conv2d(delta, w, padding=self.conf.G_kernel_size - 1)
                if ind == 0
                else F.conv2d(curr_k, w)
            )
        self.curr_k = curr_k.squeeze().flip([0, 1])

    def train(self, g_input, d_input, input_image, stat_file):
        self.set_input(g_input, d_input, input_image, stat_file)
        self.train_g()
        self.train_d()

    def set_input(self, g_input, d_input, input_image, stat_file):
        self.g_input = g_input.contiguous()
        self.d_input = d_input.contiguous()
        self.stat = stat_file
        self.input_image = input_image

    def train_g(self):
        # Zeroize gradients
        self.optimizer_G.zero_grad()
        # Generator forward pass
        g_pred = self.G.forward(self.g_input)
        # Pass Generators output through Discriminator
        d_pred_fake = self.D.forward(g_pred)
        # Calculate generator loss, based on discriminator prediction on generator result
        loss_g = self.criterionGAN(d_last_layer=d_pred_fake, is_d_input_real=True)
        # Sum all losses
        total_loss_g = loss_g + self.calc_constraints(g_pred)
        # Calculate gradients
        total_loss_g.backward()
        # Update weights
        self.optimizer_G.step()

    def calc_constraints(self, g_pred):
        # Calculate K which is equivalent to G
        self.calc_curr_k()
        # Calculate constraints
        self.loss_bicubic = self.bicubic_loss.forward(
            g_input=self.g_input, g_output=g_pred
        )
        loss_boundaries = self.boundaries_loss.forward(kernel=self.curr_k)
        loss_sum2one = self.sum2one_loss.forward(kernel=self.curr_k)
        loss_centralized = self.centralized_loss.forward(kernel=self.curr_k)
        loss_sparse = self.sparse_loss.forward(kernel=self.curr_k)
        # Apply constraints co-efficients
        return (
            self.loss_bicubic * self.lambda_bicubic
            + loss_sum2one * self.lambda_sum2one
            + loss_boundaries * self.lambda_boundaries
            + loss_centralized * self.lambda_centralized
            + loss_sparse * self.lambda_sparse
        )

    def train_d(self):
        # Zeroize gradients
        self.optimizer_D.zero_grad()
        # Discriminator forward pass over real example
        d_pred_real = self.D.forward(self.d_input)
        # Discriminator forward pass over fake example (generated by generator)
        # Note that generator result is detached so that gradients are not propagating back through generator
        g_output = self.G.forward(self.g_input)
        d_pred_fake = self.D.forward(
            (g_output + torch.randn_like(g_output) / 255.0).detach()
        )
        # Calculate discriminator loss
        loss_d_fake = self.criterionGAN(d_pred_fake, is_d_input_real=False)
        loss_d_real = self.criterionGAN(d_pred_real, is_d_input_real=True)
        loss_d = (loss_d_fake + loss_d_real) * 0.5
        # Calculate gradients, note that gradients are not propagating back through generator
        loss_d.backward()
        # Update weights, note that only discriminator weights are updated (by definition of the D optimizer)
        self.optimizer_D.step()

    def finish(self):
        final_kernel = post_process_k(self.curr_k, n=self.conf.n_filtering)
        save_final_kernel(final_kernel, self.conf)
        print("KernelGAN estimation complete!")
        return final_kernel

    def stat_calculator_save(self, image, image_2, image_4):
        stat_image = {}
        stat_image["input_mean"] = float(np.mean(image))
        stat_image["input_std"] = float(np.std(image))
        stat_image["output_2x_mean"] = float(np.mean(image_2))
        stat_image["output_2x_std"] = float(np.std(image_2))
        stat_image["output_4x_mean"] = float(np.mean(image_4))
        stat_image["output_4x_std"] = float(np.std(image_4))

        with open(self.conf.output_dir_path + "stats.json", "w") as sfile:
            json.dump(stat_image, sfile)

    def downscale(self, im, kernel, scale_factor, output_shape=None):
        """downscale function"""
        if output_shape is None:
            output_shape = np.array(im.shape[:-1]) // np.array(scale_factor)
            print(output_shape)
        # First run a correlation (convolution with flipped kernel)
        out_im = np.zeros_like(im)
        for channel in range(im.shape[-1]):
            out_im[:, :, channel] = filters.correlate(im[:, :, channel], kernel)
        width = np.round(
            np.linspace(0, im.shape[0] - scale_factor, output_shape[0])
        ).astype(int)[:, None]
        height = np.round(
            np.linspace(0, im.shape[1] - scale_factor, output_shape[1])
        ).astype(int)
        # Then subsample and return
        return out_im[width, height, :]

    def draw_images(self, kernel_image, input_image, output_image, type):
        self.figure_save(input_image, "input" + type)
        self.figure_save(kernel_image, "kernel" + type)
        self.figure_save(output_image, "output" + type)

    def figure_save(self, image, type):
        figure = plt.figure()
        if len(image.shape) > 2:
            image = image[:, :, 0]
        m = np.mean(image)
        s = np.std(image)
        plt.imsave(
            self.conf.output_dir_path + "/" + self.conf.img_name + type + ".png",
            image,
            cmap="gray",
            vmin=self.stat["min"],
            vmax=m + 3 * s,
        )
        # figure.savefig(self.conf.output_dir_path +"/" + self.conf.img_name + type+'.png')

    def re_normalize(self, image):
        image = (image * self.stat["std"]) + self.stat["mean"]
        return image
