#!/usr/bin/env python3

"""Usage:   trainer.py --train=train_path --valid=valid_path --log_dir=log_dir --num_epochs=epochs --architecture=arch --act=act --kernel_factor=factor --model_save=model[--lognorm] [--debug_input_pics] [--aspp] [--dilation]
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

import torch
import torch.optim as optim
import torch.nn

from torchsummary import summary

from docopt import docopt

import numpy as np
import tifffile
import matplotlib.pyplot as plt
from tqdm import tqdm
from unet import UNET
from edsr import EDSR
from dataset import SrDataset, PairedDataset
from axial_bicubic import AxialNet
from losses import SSIM, L1loss, PSNR
from logger import Logger

BATCH_SIZE = {
    "unet": 16,
    "axial": 16,
    "edsr_16_64": 8,
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


def model_save(train_model, train_model_path):
    """

    Parameters
    ----------
    train_model: model state dictionary
    train_model_path: model path

    Returns
    -------

    """
    model_path = Path(train_model_path)
    model_folder = model_path.parent
    if not model_folder.is_dir():
        os.makedirs(model_folder)
    torch.save(train_model.state_dict(), model_path)


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
    model_save_path
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
    if not save_model_path.is_dir():
        os.makedirs(save_model_path)
    save_model_path = str(save_model_path)
    # parameters
    lr = LR[architecture]
    if architecture == "unet":
        model = UNET(in_channels=1, out_channels=1, init_features=32)
    elif architecture == "axial":
        model = AxialNet(num_channels=1, resblocks=2, skip=1)
    elif architecture == "edsr_16_64":
        model = EDSR(
            n_resblocks=16, n_feats=64, scale=4, aspp=aspp, dilation=dilation, act=act
        )
    elif architecture == "edsr_8_256":
        model = EDSR(
            n_resblocks=8, n_feats=256, scale=4, aspp=aspp, dilation=dilation, act=act
        )
    elif architecture == "edsr_16_256":
        model = EDSR(
            n_resblocks=16, n_feats=256, scale=4, aspp=aspp, dilation=dilation, act=act
        )
    elif architecture == "edsr_32_256":
        model = EDSR(n_resblocks=32, n_feats=256, scale=4)
    model.to(device)
    summary(model, (1, 32, 32), batch_size=1, device="cuda")
    max_epochs = num_epochs
    criterion = torch.nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # learning rate scheduler
    if architecture == "edsr":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    best_valid_loss = float("inf")
    logger = Logger(str(log_dir))
    step = 0
    # TODO: Remove after debugging is done
    if debug_pics:
        input_save_path = Path(os.path.dirname(__file__) + r"/input_pics")
        if input_save_path.is_dir():
            shutil.rmtree(input_save_path)
        os.makedirs(input_save_path)
    for epoch in range(max_epochs):
        start_time = time()
        train_loss = valid_loss = 0.0
        model.train()
        loss_train_list = []
        step += 1
        # Main training loop for this epoch
        for batch_idx, data in enumerate(tqdm(training_generator)):
            model.train(True)
            x_train = data["lr"]
            y_train = data["hr"]
            stat = data["stats"]
            mean, sigma = stat["mean"], stat["std"]
            x_train, y_train, mean, sigma = (
                x_train.to(device),
                y_train.to(device),
                mean.to(device),
                sigma.to(device),
            )
            # TODO: Remove this after debugging is over
            if debug_pics:
                x_np = x_train.cpu().numpy()
                y_np = y_train.cpu().numpy()
                for i in range(x_np.shape[0]):
                    filename = data["file"][i]
                    x_rescale = (
                        x_np[i] * stat["std"][i].numpy() + stat["mean"][i].numpy()
                    )
                    y_rescale = (
                        y_np[i] * stat["std"][i].numpy() + stat["mean"][i].numpy()
                    )

                    # image padded to make sure lr and hr are same size
                    x_rescale_pad = x_rescale.reshape(x_rescale.shape[1], -1)
                    image_width, image_height = x_rescale_pad.shape
                    image_width = (image_width // 2) * 3
                    image_height = (image_height // 2) * 3
                    x_rescale_pad = np.pad(x_rescale_pad, [image_width, image_height])
                    save_plots = np.hstack(
                        [
                            x_rescale_pad,
                            y_rescale.reshape(y_rescale.shape[1], -1),
                        ]
                    )
                    save_plots = np.clip(
                        save_plots, stat["min"][i].numpy(), stat["max"][i].numpy()
                    )
                    vmax = stat["mean"][i].numpy() + 3 * stat["std"][i].numpy()
                    vmin = stat["min"][i].numpy()
                    filename = os.path.join(f"{input_save_path}/{filename}.tiff")
                    plt.imsave(filename, save_plots, vmin=vmin, vmax=vmax, cmap="gray")

            optimizer.zero_grad()
            with torch.autograd.set_detect_anomaly(True):
                with torch.set_grad_enabled(True):
                    y_pred = model(x_train)
                    loss_train = criterion(y_pred, y_train)
                    train_loss = train_loss + (
                        (1 / (batch_idx + 1)) * (loss_train.data - train_loss)
                    )
                    loss_train_list.append(loss_train.item())
                    loss_train.backward()
                    optimizer.step()

        # training log summary after every 10 epochs
        log_loss_summary(logger, loss_train_list, step, prefix="train_")
        loss_train_list = []

        del x_train, y_train, mean, sigma, loss_train_list
        torch.cuda.empty_cache()

        # Main validation loop for this epoch
        if architecture == "edsr":
            scheduler.factor = 1 + (epoch / max_epochs) ** 0.9
        with torch.no_grad():
            loss_valid_list = []
            for batch_idx, data in enumerate(tqdm(validation_generator)):
                # unet.eval()
                model.train(False)
                x_valid = data["lr"]
                y_valid = data["hr"]

                x_valid, y_valid = x_valid.to(device), y_valid.to(device)
                y_pred = model(x_valid)
                loss_valid = criterion(y_pred, y_valid)
                loss_valid_list.append(loss_valid.item())
                valid_loss = valid_loss + (
                    (1 / (batch_idx + 1)) * (loss_valid.data - valid_loss)
                )

        if architecture == "edsr":
            # calling scheduler based on valid loss
            scheduler.step(valid_loss)
            # print(optimizer.param_groups[0]['lr'])
        # valid log summary after every 10 epochs
        log_loss_summary(logger, loss_valid_list, step, prefix="val_")
        loss_valid_list = []

        del x_valid, y_valid, loss_valid_list
        memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        print(
            "\nEpoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} in {:.1f} seconds. [lr:{:.8f}][max mem:{:.0f}MB]".format(
                epoch,
                train_loss,
                valid_loss,
                time() - start_time,
                optimizer.param_groups[0]["lr"],
                memory,
            )
        )
        # Save best validation epoch model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            model_save(
                model, f"{save_model_path}/{architecture}/{timestamp}_best_model.pt",
            )

        if step % 10 == 0:
            model_save(
                model, f"{save_model_path}/{architecture}/{timestamp}_model_{step}.pt",
            )
        model_save(model, f"{save_model_path}/{architecture}/{timestamp}_model.pt")
        torch.cuda.empty_cache()


def process(train_path, valid_path, log_dir, architecture, num_epochs, lognorm, debug_pics, aspp,
            dilation, act, model_save_path):
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
        model_save_path
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
    kernel_factor = arguments['--kernel_factor']
    model_save_path = Path(arguments['--model_save'])

    process(train_path, valid_path, log_dir, architecture, num_epochs, lognorm, debug_pics, aspp, dilation, act, model_save_path, kernel_factor)
