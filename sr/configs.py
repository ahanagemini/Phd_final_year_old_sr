import argparse
import torch
import itertools
from pathlib import Path
import os


# noinspection PyPep8
class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.conf = None
        self.image = None
        self.stats = None

        # kernel_Gan or PIl_resize
        self.parser.add_argument(
            "--kernel_gan",
            type=bool,
            default=None,
            help="Enable this command to use kernel gan",
        )

        # Paths
        self.parser.add_argument(
            "--img_name", default="image1", help="image name for saving purposes"
        )
        self.parser.add_argument(
            "--input_dir_path",
            default=os.path.dirname(__file__) + r"/input",
            help="all inputs are in this path",
        )
        self.parser.add_argument(
            "--cutting_output_dir_path",
            default=os.path.dirname(__file__) + r"/cutter_out",
            help="here cut images will be stored",
        )
        self.parser.add_argument(
            "--output_dir_path",
            default=os.path.dirname(__file__) + r"/results",
            help="results path",
        )
        self.parser.add_argument(
            "--stat_image_path",
            default=os.path.dirname(__file__) + r"/training_data/stats.json",
            help="path to the stat file of image",
        )

        # Sizes
        self.parser.add_argument(
            "--input_crop_size", type=int, default=64, help="Generators crop size"
        )
        self.parser.add_argument(
            "--scale_factor",
            type=float,
            default=0.25,
            help="The downscaling scale factor",
        )
        self.parser.add_argument(
            "--X4", action="store_true", help="The wanted SR scale factor"
        )

        # Network architecture
        self.parser.add_argument(
            "--G_chan",
            type=int,
            default=64,
            help="# of channels in hidden layer in the G",
        )
        self.parser.add_argument(
            "--D_chan",
            type=int,
            default=64,
            help="# of channels in hidden layer in the D",
        )
        self.parser.add_argument(
            "--G_kernel_size",
            type=int,
            default=13,
            help="The kernel size G is estimating",
        )
        self.parser.add_argument(
            "--D_n_layers", type=int, default=7, help="Discriminators depth"
        )
        self.parser.add_argument(
            "--D_kernel_size",
            type=int,
            default=7,
            help="Discriminators convolution kernels size",
        )

        # Iterations
        self.parser.add_argument(
            "--max_iters", type=int, default=10000, help="# of iterations"
        )

        # Optimization hyper-parameters
        self.parser.add_argument(
            "--g_lr",
            type=float,
            default=2e-4,
            help="initial learning rate for generator",
        )
        self.parser.add_argument(
            "--d_lr",
            type=float,
            default=2e-4,
            help="initial learning rate for discriminator",
        )
        self.parser.add_argument(
            "--beta1", type=float, default=0.5, help="Adam momentum"
        )

        # Extra Kernel Gan Parameters
        self.parser.add_argument(
            "--kernel_save",
            default=os.path.dirname(__file__) + r"/kernel_folder",
            help="The kernel files will be stored here",
        )
        self.parser.add_argument(
            "--compare",
            type=bool,
            default=False,
            help="plots to images for comparision",
        )

        self.parser.add_argument(
            "--save_compare_stat",
            type=bool,
            default=False,
            help="save image stat for comparision",
        )

        # Trainer.py Parameters
        self.parser.add_argument(
            "--train",
            default=os.path.dirname(__file__) + r"/results/train",
            help="train files path",
        )
        self.parser.add_argument(
            "--valid",
            default=os.path.dirname(__file__) + r"/results/valid",
            help="valid files path",
        )
        self.parser.add_argument(
            "--log_dir",
            default=os.path.dirname(__file__) + r"/logger",
            help="the log files will be stored in this directory",
        )
        self.parser.add_argument(
            "--architecture", default="edsr_16_64", help="give the model to be train"
        )
        self.parser.add_argument(
            "--num_epochs", type=int, default=100, help="the total number of epochs"
        )
        self.parser.add_argument(
            "--lognorm",
            type=bool,
            default=False,
            help="check whether lognorm is required or not",
        )
        self.parser.add_argument(
            "--debug_pics",
            type=bool,
            default=False,
            help="check if debug pics are required",
        )
        self.parser.add_argument(
            "--aspp", type=bool, default=False, help="check if edsr needs aspp"
        )
        self.parser.add_argument(
            "--dilation", type=bool, default=False, help="check if edsr needs dilation"
        )
        self.parser.add_argument(
            "--act",
            default="leakyrelu",
            help="activation type relu or leakyrelu for edsr",
        )
        self.parser.add_argument(
            "--model_save",
            default=os.path.dirname(__file__) + "r/saved_models",
            help="the path where model will be saved",
        )
        self.parser.add_argument(
            "--load_last_trained",
            type=bool,
            default=False,
            help="This command is to can be used to load the last model trained",
        )
        self.parser.add_argument(
            "--resume",
            default=None,
            help="this coomand can be used to load trained model",
        )

        # Tester.py Parameters
        self.parser.add_argument("--test_input_dir")
        self.parser.add_argument(
            "--active",
            type=bool,
            default=False,
            help="Whether to save per-image metrics for active learning selection",
        )
        self.parser.add_argument(
            "--save_slice",
            type=bool,
            default=False,
            help="If we want to save the slice as a png image",
        )

        # GPU
        self.parser.add_argument("--gpu_id", type=int, default=0, help="gpu id number")

        # Kernel post processing
        self.parser.add_argument(
            "--n_filtering",
            type=float,
            default=40,
            help="Filtering small values of the kernel",
        )

        # Number of resize operations
        self.parser.add_argument(
            "--n_resize",
            type=int,
            default=10,
            help="Number of resize operations/scales.",
        )
        self.parser.add_argument(
            "--noise_scale",
            type=float,
            default=1.0,
            help="ZSSR uses this to partially de-noise images",
        )
        self.parser.add_argument(
            "--real_image",
            action="store_true",
            help="ZSSRs configuration is for real images",
        )

    def parse(self, args=None):
        """Parse the configuration"""
        self.conf = self.parser.parse_args(args=args)
        self.set_gpu_device()
        self.conf.G_structure = [7, 5, 3, 1, 1, 1]

        # if model available which needs to resume training
        if self.conf.resume:
            self.conf.model_save = self.conf.resume
            return self.conf

        # if no model has been created yet
        self.conf.model_save = self.conf.model_save + f"/{self.conf.architecture}"
        if not os.path.isdir(self.conf.model_save):
            os.makedirs(self.conf.model_save)

        self.conf.model_save = self.get_model_directories(self.conf.model_save)

        #  if command not present it will create new directory
        #  for saving models or else last model trained will be loaded
        if not self.conf.load_last_trained:
            #self.conf.model_save = self.check_model_directory(self.conf.model_save)
            if not os.path.isdir(self.conf.model_save):
                 os.makedirs(self.conf.model_save)
        return self.conf

    def get_model_directories(self, model_save):
        """

        Parameters
        ----------
        model_save:

        Returns
        -------
        last_folder
        """
        directories = os.scandir(model_save)
        directories = [str(x.path) for x in directories]
        if not directories:
            return model_save + f"/{self.conf.architecture}"
        last_folder = sorted(directories)[-1]
        return last_folder

    def check_model_directory(self, model_save):
        """

        Parameters
        ----------
        model_save: Path of model directory

        Returns
        -------
        model_save
        """
        if os.path.isdir(model_save):
            for i in itertools.count(start=1, step=1):
                new_folder = model_save + "({0})".format(i)
                if not os.path.isdir(new_folder):
                    model_save = new_folder
                    break
        return model_save

    def set_gpu_device(self):
        """Sets the GPU device if one is given"""
        if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.conf.gpu_id)
            torch.cuda.set_device(0)
        else:
            torch.cuda.set_device(self.conf.gpu_id)
