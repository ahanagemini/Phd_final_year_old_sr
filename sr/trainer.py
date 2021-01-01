#!/usr/bin/env python3

"""Usage:   trainer.py --train=train_path --valid=valid_path --log_dir=log_dir --num_epochs=epochs --architecture=arch --act=act --kernel_factor=factor --model_save=model[--lognorm] [--debug_input_pics] [--aspp] [--dilation] [--resume]
            trainer.py --help | -help | -h

Train the requested model.
Arguments:
  train         a directory with training images/ direcrtories/ numpy
  output        a directory for validation images/ directories/ numpy
  log_dir       directory for storing training logs
  num_epochs    number of epochs
  architecture  the architecture to train unet or axial
  act activations can be relu or leakyrelu FOR EDSR ONLY
  kernel_factor use if using kernel, for 0.25 use --X4. for 0.5 use --X2 for 0.125 use --X8
  model_save    The path where model will be saved while training
  --lognorm     if we are using log normalization
  --debug_input_pics  If we want to save input pics for debugging
  --aspp        use ASPP in EDSR
  --dilation    use dilation at the beginning in edsr
  --resume      set this if you want resume a model
Options:
  -h --help -h
"""

# Example run :
#  python3.8 ./trainer.py --train=../earth_data/train/f3 --valid=../earth_data/valid/f3 --log_dir=logs --num_epochs=100 --architecture=edsr_16_64 --act=relu

from pathlib import Path
import os
import sys
import shutil
from time import time
import datetime
import scipy.ndimage

import torch
import torch.optim as optim
import torch.nn

from torchsummary import summary

from docopt import docopt

import numpy as np
import tifffile
import matplotlib.pyplot as plt
from tqdm import tqdm
from models import UNET
from models import EDSR
from dataset import SrDataset, PairedDataset
from axial_bicubic import AxialNet
from losses import SSIM, L1loss, PSNR, Column_Difference, Row_Difference
from logger import Logger
from train_util import model_selection, debug_pics,  check_load_model, model_save, train_model, valid_model

BATCH_SIZE = {
    "unet": 4,
    "axial": 16,
    "edsr_16_64": 16,
    "edsr_8_256": 16,
    "edsr_16_256": 8,
    "edsr_32_256": 8,
}
LR = {
    "unet": 0.00005,
    "axial": 0.0005,
    "edsr_16_64": 0.0005,
    "edsr_8_256": 0.0001,
    "edsr_16_256": 0.0001,
    "edsr_32_256": 0.0001,
}


def log_loss_summary(logger, loss, step, prefix=""):
    logger.scalar_summary(prefix + "loss", np.mean(loss), step)
def model_draw(logger, model, input_to_model):
    logger.model_graph(model, input_to_model)

def create_dataset(path, lognorm=False):
    """

    Parameters
    ----------
    path: path to data directory
    lognorm: Is log normalization used?
    kernel: Whether to use kernel or not default false
    kernel_factor: if using kernel then how much to factor the image default --X4 (0.25)

    Returns
    -------
    Loaded dataset

    """

    if set(os.listdir(path)) == set(["LR", "HR"]):
        print("running PairedDataset")
        return PairedDataset(path, lognorm=lognorm)
    else:
        print("Running SrDataset")
        return SrDataset(path, lognorm=lognorm)


def training(
    training_generator,
    validation_generator,
    device,
    log_dir,
    architecture,
    num_epochs,
    debug_pics,
    aspp,
    dilation,
    act,
    model_save_path,
    kernel,
):
    """

    Parameters
    ----------
    training_generator: contains training data
    validation_generator: contains validation data
    device: Cuda Device
    log_dir: The log directory for storing logs
    architecture: The architecture to be used unet or axial
    num_epochs:   The number of epochs
    debug_pics: True if we want to save pictures in input_pics
    aspp: True if EDSR is going to use aspp
    dilation True if EDSR is going to use dilation
    act: activation function to be used
    model_save: location where model will be saved
    Returns
    -------

    """
    timestamp = f"{datetime.datetime.now().date()}-{datetime.datetime.now().time()}"
    save_model_path = Path(model_save_path)
    current_save_model_path = save_model_path / "current"
    best_save_model_path = save_model_path / "best"
    logger = Logger(str(log_dir))

    # setting up the srchitecture for training
    model = model_selection(architecture, aspp, dilation, act)
    # parameters
    lr = LR[architecture]
    model.to(device)
    summary(model, (1, 64, 64), batch_size=1, device="cuda")
    max_epochs = num_epochs

    # drawing model
    dummy_input = torch.from_numpy(np.random.randn(1, 1, 64, 64)).float()
    dummy_input = dummy_input.to(device)
    model_draw(logger, model, dummy_input)
    del dummy_input

    # loading the model
    training_parameters = check_load_model(save_model_path, model, lr)
    training_parameters["criterion"] = torch.nn.L1Loss()
    training_parameters["row_diff_loss"] = Row_Difference()
    training_parameters["device"] = device
    training_parameters["lambda_row"] = 0.1
    training_parameters["max_epochs"] = max_epochs

    best_valid_loss = float("inf")

    step = training_parameters["current_epoch"]
    # TODO: Remove after debugging is done
    if debug_pics:
        input_save_path = Path(os.path.dirname(__file__) + r"/input_pics")
        if input_save_path.is_dir():
            shutil.rmtree(input_save_path)
        os.makedirs(input_save_path)

    while step < max_epochs:
        print(f"current step is {step}")
        start_time = time()
        model.train()
        # training
        loss_train_list, train_loss, l1_loss, row_loss, l1_list, row_diff_list = train_model(training_generator, training_parameters)

        # training log summary after every 10 epochs
        log_loss_summary(logger, loss_train_list, step, prefix="train_")
        log_loss_summary(logger, l1_list, step, prefix="_train_l1")
        log_loss_summary(logger, row_diff_list, step, prefix="_train_row")
        del loss_train_list, l1_list, row_diff_list
        torch.cuda.empty_cache()

        # validation
        loss_valid_list, valid_loss, valid_l1_loss, valid_row_loss, val_l1_list, val_row_list = valid_model(validation_generator, training_parameters)
        log_loss_summary(logger, loss_valid_list, step, prefix="val_")
        log_loss_summary(logger, val_l1_list, step, prefix="_val_l1")
        log_loss_summary(logger, val_row_list, step, prefix="_val_row")
        del loss_valid_list, val_l1_list, val_row_list
        torch.cuda.empty_cache()


        memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        print(
            "\nEpoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} in {:.1f} seconds. [lr:{:.8f}][max mem:{:.0f}MB]".format(
                training_parameters["current_epoch"],
                train_loss,
                valid_loss,
                time() - start_time,
                training_parameters["optimizer"].param_groups[0]["lr"],
                memory,
            ))

        print("Train_L1_loss: {:.6f} \t Train_Row loss: {:.6f} \t Valid_L1_loss: {:.6f} \t Valid_row_loss: {:.6f}".format(
            l1_loss, row_loss, valid_l1_loss, valid_row_loss)
        )

        save_params = {
                "epoch": step,
                "model": model.state_dict(),
                "optimizer": training_parameters["optimizer"].state_dict(),
                "scheduler": training_parameters["scheduler"].state_dict(),
        }
        # Save best validation epoch model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

            model_save(save_params, f"{str(best_save_model_path)}/best_model.pt")
            # deleting the old_best_model after new one is saved
            if os.path.isfile(training_parameters["best_model"]):
                os.remove(training_parameters["best_model"])

            current_best_model_list = list(best_save_model_path.rglob("*.pt"))
            if not current_best_model_list:
                training_parameters["best_model"] = ""
            else:
                training_parameters["best_model"] = current_best_model_list[-1]

        model_save(
            save_params, f"{str(current_save_model_path)}/{timestamp}_model_{step}.pt"
        )

        # deleting old current model after new one is saved
        print(f"the current model path is {training_parameters['current_model']}")
        if os.path.isfile(training_parameters["current_model"]):
            os.remove(training_parameters["current_model"])

        current_model_list = list(current_save_model_path.rglob("*.pt"))
        if not current_model_list:
            training_parameters["current_model"] = ""
        else:
            training_parameters["current_model"] = current_model_list[-1]

        torch.cuda.empty_cache()
        step += 1
        training_parameters["current_epoch"] = step


def process(
    train_path,
    valid_path,
    log_dir,
    architecture,
    num_epochs,
    lognorm,
    debug_pics,
    aspp,
    dilation,
    act,
    model_save_path,
    kernel=False,
):
    """

    Parameters
    ----------
    arguments: all the argumennts passed to trainer.py

    Returns
    -------

    """

    parameters = {
        "batch_size": BATCH_SIZE[architecture],
        "shuffle": True,
        "num_workers": 6,
    }

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    training_set = create_dataset(train_path, lognorm=lognorm)
    validation_set = create_dataset(valid_path, lognorm=lognorm)

    training_generator = torch.utils.data.DataLoader(training_set, **parameters)
    validation_generator = torch.utils.data.DataLoader(validation_set, **parameters)

    training(
        training_generator,
        validation_generator,
        device,
        log_dir,
        architecture,
        num_epochs,
        debug_pics,
        aspp,
        dilation,
        act,
        model_save_path,
        kernel,
    )


if __name__ == "__main__":
    arguments = docopt(__doc__)

    print("Processing arguments...")
    train_path = Path(arguments["--train"])
    valid_path = Path(arguments["--valid"])
    log_dir = Path(arguments["--log_dir"])
    architecture = arguments["--architecture"]
    num_epochs = int(arguments["--num_epochs"])
    lognorm = arguments["--lognorm"]
    debug_pics = arguments["--debug_input_pics"]
    aspp = arguments["--aspp"]
    dilation = arguments["--dilation"]
    act = arguments["--act"]
    model_save_path = Path(arguments["--model_save"])

    process(
        train_path,
        valid_path,
        log_dir,
        architecture,
        num_epochs,
        lognorm,
        debug_pics,
        aspp,
        dilation,
        act,
        model_save_path,
    )
