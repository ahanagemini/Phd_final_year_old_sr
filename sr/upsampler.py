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
from train_util import chop_forward, forward_chop, model_selection


def prepare(tensor, conf):
    """

    Parameters
    ----------
    tensor
    conf

    Returns
    -------

    """
    if conf.precision == "half":
        tensor = tensor.half()
    return tensor


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
    return Upsampler_Dataset(path, lognorm)


def upsampler(conf):
    """
    The main function that reads configuration and upsamples a directory
    """
    parameters = {"batch_size": 1, "shuffle": False, "num_workers": 6}
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    idir = Path(conf.input)
    if not os.path.isdir(Path(conf.output)):
        os.makedirs(Path(conf.output))

    test_set = create_dataset(idir, conf.lognorm)
    test_generator = torch.utils.data.DataLoader(test_set, **parameters)
    model = model_selection(conf.architecture, conf.aspp, conf.dilation, conf.act)
    model = model.to(device)
    checkpoint = torch.load(conf.model)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    if conf.precision == "half":
        model.half()
    with torch.no_grad():
        for i, sample in enumerate(tqdm(test_generator)):
            stats = sample["stats"]
            mean = stats["mean"]
            std = stats["std"]
            mean = mean.to(device)
            std = std.to(device)
            filename = str(i) + ".png"
            filepath = Path(conf.output) / filename
            sample["lr"] = prepare(sample["lr"], conf)

            # The actual upsample happens here
            y_pred = forward_chop(
                sample["lr"].to(device), model=model, scale=4, shave=32, min_size=16384
            )
            y_pred = (y_pred * std) + mean
            y_pred = np.clip(
                y_pred.cpu(), stats["min"].cpu().numpy(), stats["max"].cpu().numpy()
            )
            if conf.lognorm:
                image_sign = np.sign(y_pred)
                y_pred = image_sign * (np.exp(np.abs(y_pred)) - 1.0)
                del image_sign
            y_pred = y_pred.reshape(-1, y_pred.shape[-1])
            y_pred = y_pred.numpy()

            height, width = y_pred.shape
            y_pred = y_pred[sample["top"] * 4: height - sample["bottom"] * 4,
                     sample["left"] * 4:width - sample["right"] * 4]

            vmax = np.max(y_pred)
            vmin = np.min(y_pred)
            plt.imsave(str(filepath), y_pred, cmap="gray", vmax=vmax, vmin=vmin)

            del y_pred, vmax, vmin, filepath, stats, filename


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
            "--output",
            default=os.path.dirname(os.path.abspath(__file__)) + r"/output_dir",
            help="use this command to set output directory",
        )
        self.parser.add_argument(
            "--model",
            default=os.path.dirname(os.path.abspath(__file__)) + r"/model_dir",
            help="use this command to set model directory",
        )
        self.parser.add_argument(
            "--architecture",
            default="edsr_16_64",
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
        self.parser.add_argument(
            "--precision",
            default="half",
            help="use this to set the command to change the precision",
        )

    def parse(self, args=None):
        """Parse the configuration"""
        self.conf = self.parser.parse_args(args=args)
        return self.conf


if __name__ == "__main__":
    conf = Configurator().parse()
    upsampler(conf)
