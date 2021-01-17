
"""
This module will be used to upsample input images.
"""
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import torch
from GPUtil import showUtilization as gpu_usage
from torch.backends import cudnn
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from tqdm import tqdm

from cutter import loader
from stat_plotter import PlotStat
from test_dataset import Upsampler_Dataset
from train_util import model_selection
from vgg_dataset import VGGTrainDataset

def create_dataset(path, lognorm=False):
    """

    Parameters
    ----------
    path: path to data directory
    lognorm: Is log normalization used?

    Returns
    -------
    Loaded dataset

    """
    return VGGTrainDataset(path, lognorm)


def vgg_testing(conf):
    """
    The main function that reads configuration and upsamples a directory
    """
    parameters = {"batch_size": 1, "shuffle": False, "num_workers": 6}
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    idir = Path(conf.input)
    test_set = create_dataset(idir, conf.lognorm)
    test_generator = torch.utils.data.DataLoader(test_set, **parameters)
    model = model_selection(conf.architecture, conf.aspp, conf.dilation, conf.act)
    model = model.to(device)
    checkpoint = torch.load(conf.model_save)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(tqdm(test_generator)):
            # The actual upsample happens here
            y_pred = model.forward(sample["image"].to(device))
            print(y_pred)
            del y_pred


class Configurator:
    """ This is the config class for tester"""

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            "--input",
            default=os.path.dirname(os.path.abspath(__file__)) + r"/input_dir",
            help="use this command to set input directory",
        )
        self.parser.add_argument(
            "--model_save",
            default=os.path.dirname(os.path.abspath(__file__)) + r"/model_dir",
            help="use this command to set model directory",
        )
        self.parser.add_argument(
            "--architecture",
            default="vgg",
            help="use this command to set architecture",
        )
        self.parser.add_argument(
            "--act", default="leakyrelu", help="use this command to set activation"
        )
        self.parser.add_argument(
            "--aspp", default=False, help="use this to set aspp for edsr"
        )
        self.parser.add_argument(
            "--dilation", default=False, help="use this to set dilation for edsr"
        )
        self.parser.add_argument(
            "--lognorm", default=False, help="use this command to set lognorm"
        )

    def parse(self, args=None):
        """Parse the configuration"""
        self.conf = self.parser.parse_args(args=args)
        return self.conf


if __name__ == "__main__":
    conf = Configurator().parse()
    vgg_testing(conf)
