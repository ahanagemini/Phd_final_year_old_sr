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

    test_set = create_dataset(args["--input"], lognorm=args["--lognorm"], test=True, hr=args["hr"])
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
    


if __name__ == "__main__":
    '''
    args = docopt(__doc__)
    for a in ["--input", "--output", "--model"]:
        args[a] = Path(args[a]).resolve()
    print("Input being read from:", args["--input"])
    '''

    args = {'--help': False,
     '--input': True,
     '--model': True,
     '--output': True,
     '-e': False,
     '-l': False,
     '-p': False,
     '--input': r"/home/venkat/Documents/PiyushKumarProject/Libraries/predict",
     '--model': r'/home/venkat/Documents/PiyushKumarProject/Libraries/edsr_16_64/edsr_16_64/current/2020-12-28-21:20:05.167183_model_200.pt',
     '--output': r'/home/venkat/Documents/PiyushKumarProject/Libraries/predict_output',
     '--architecture': 'edsr_16_64',
     'hr': True,
     'lr': False,
     '--act': 'leakyrelu',
     '--dilation': False,
     '--aspp': False,
     '--lognorm': False,
     '--active': False,
     'kernel': True,
     "--save_slice": False}
    evaluate(args)
