#!/usr/bin/env python3

"""Usage:   tester.py --input=input --output=output --model=model --architecture=arch
                           --act=act [--lognorm] [--active] [--save_slice] [--aspp] [--dilation]
            tester.py --help | -help | -h

Process input to create super resolution using the model input and plot it. In hr mode, the input is assumed to be high resolution. In lr mode the input is assumed to be low resolution.
Arguments:
  input         a directory with input files
  output        a directory for saving the output images
  model         a .pt file to use for the model
  architecture  architecture to use: unet or axial
  act           activation used in EDSR 'relu' or 'leakyrelu'
  --lognorm     If we want to use log normalization
  --active      Whether to save per-image metrics for active learning selection
  --save_slice  If we want to save the slice as a png image
  --dilation    Use dilation in head convolutions for EDSR
  --aspp        Use aspp at the end of the model in EDSR

Options:
  -h --help -h
"""
import os
from pathlib import Path
import shutil
import torch

import tifffile
from dataset import SrDataset, PairedDataset
from docopt import docopt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import metrics
from models import UNET
from axial_bicubic import AxialNet
from models import EDSR
from tqdm import tqdm
from train_util import model_selection, test_model
import scipy.ndimage
import argparse

from PIL import Image, ImageFont, ImageDraw



def create_dataset(path, lognorm=False, test=True, hr=True):
    """

    Parameters
    ----------
    path: path to data directory
    lognorm: Is log normalization used?

    Returns
    -------
    Loaded dataset

    """

    if set(os.listdir(path)) == set(["LR", "HR"]):
        return PairedDataset(path, lognorm=lognorm, test=test)
    else:
        return SrDataset(path, lognorm=lognorm, test=test, hr=hr)

def evaluate(args):
    """

    Parameters
    ----------
    args: the docopt arguments passed

    Returns
    -------

    """
    parameters = {"batch_size": 1, "shuffle": False, "num_workers": 6}
    use_cuda = torch.cuda.is_available()
    lognorm = args["--lognorm"]
    active_learning = args["--active"]
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    test_set = create_dataset(args["--input"], lognorm=args["--lognorm"], test=True, vaidate=False, hr=args["hr"])
    test_generator = torch.utils.data.DataLoader(test_set, **parameters)
    dilation = args["--dilation"]
    aspp = args["--aspp"]
    act = args["--act"]
    # model create and load
    model = model_selection(args["--architecture"], aspp, dilation, act)

    model.to(device)
    checkpoint = torch.load(args["--model"])
    model.load_state_dict(checkpoint['model'])
    testing_parameters = {
        "model" : model,
        "args" : args,
        "lognorm" : args["--lognorm"],
        "device" : device,
        "active" : active_learning
    }
    test_model(test_generator, testing_parameters)
    

class Configurator:
    """ This is the config class for tester"""
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--input', default=os.path.dirname(os.path.abspath(__file__)) +r"/input_dir",
                                 help="use this command to set input directory")
        self.parser.add_argument('--output', default=os.path.dirname(os.path.abspath(__file__)) + r"/output_dir",
                                 help="use this command to set output directory")
        self.parser.add_argument('--model', default=os.path.dirname(os.path.abspath(__file__)) +r"/model_dir",
                                 help="use this command to set model directory")
        self.parser.add_argument('--architecture', default="edsr_16_64",
                                 help="use this command to set architecture")
        self.parser.add_argument('--act', default="leakyrelu",
                                 help="use this command to set activation")
        self.parser.add_argument('--aspp', default=False,
                                 help="use this to set aspp for edsr")
        self.parser.add_argument('--dilation', default=False,
                                 help="use this to set dilation for edsr")
        self.parser.add_argument('--hr', default=True,
                                 help="use this command to set hr")
        self.parser.add_argument('--lr', default=False,
                                 help="use this command to set lr")
        self.parser.add_argument('--lognorm', default=False,
                                 help="use this command to set lognorm")
        self.parser.add_argument('--active', default=False,
                                 help="use this command to set active")


    def parse(self, args=None):
        """Parse the configuration"""
        self.conf = self.parser.parse_args(args=args)
        return self.conf

if __name__ == "__main__":
    conf = Configurator().parse()
    args = {
     '--input': conf.input,
     '--model': conf.model,
     '--output': conf.output,
     '--architecture':  conf.architecture,
     'hr': conf.hr,
     'lr': conf.lr,
     '--act': conf.act,
     '--dilation': conf.dilation,
     '--aspp': conf.aspp,
     '--lognorm': conf.lognorm,
     '--active': conf.active,
     'kernel': True,
     "--save_slice": False}
    evaluate(args)
