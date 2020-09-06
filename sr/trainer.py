#!/usr/bin/env python3

"""Usage:   trainer.py --train=train_path --valid=valid_path --log_dir=log_dir --architecture=arch
            trainer.py --help | -help | -h

Train the requested model.
Arguments:
  train         a directory with training images/ direcrtories/ numpy
  output        a directory for validation images/ directories/ numpy
  log_dir       directory for storing training logs
  architecture  the architecture to train unet or axial
Options:
  -h --help -h
"""

from pathlib import Path
import os
from time import time

import torch
import torch.optim as optim
import torch.nn

from torchsummary import summary

from docopt import docopt

import numpy as np
from tqdm import tqdm
from unet import UNET
from dataset import SrDataset
from axial_bicubic import AxialNet
from losses import SSIM, L1loss, PSNR
from logger import Logger


def log_loss_summary(logger, loss, step, prefix=""):
    logger.scalar_summary(prefix + "loss", np.mean(loss), step)


def training(training_generator, validation_generator, device, log_dir, architecture):
    """

    Parameters
    ----------
    training_generator: contains training data
    validation_generator: contains validation data
    device: Cuda Device
    log_dir: The log directory for storing logs
    architecture: The architecture to be used unet or axial

    Returns
    -------

    """
    # parameters
    if architecture == "unet":
        model = UNET(in_channels=1, out_channels=1, init_features=32)
    elif architecture == "axial":
        model = AxialNet(num_channels=1, resblocks=2, skip=1)
    model.to(device)
    summary(model, (1, 256, 256), batch_size=-1, device="cuda")
    max_epochs = 200
    # criterion = SSIM()
    # criterion = PSNR()
    # criterion = L1loss()
    criterion = torch.nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    best_valid_loss = float("inf")
    logger = Logger(str(log_dir))
    step = 0
    totiter = sum(1 for x in training_generator)
    valiter = sum(1 for x in validation_generator)

    for epoch in range(max_epochs):
        start_time = time()
        train_loss = valid_loss = 0.0
        model.train()
        loss_train_list = []
        step += 1
        # Main training loop for this epoch
        for batch_idx, data in tqdm(enumerate(training_generator), total=totiter):
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
        scheduler.factor = 1 + (epoch / max_epochs) ** 0.9
        with torch.no_grad():
            loss_valid_list = []
            for batch_idx, data in tqdm(enumerate(validation_generator), total=valiter):
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
            torch.save(
                model.state_dict(), f"{os.getcwd()}/{architecture}_best_model.pt"
            )

        if step % 10 == 0:
            torch.save(
                model.state_dict(), f"{os.getcwd()}/{architecture}_model_{step}.pt"
            )
        torch.save(model.state_dict(), f"{os.getcwd()}/{architecture}_model.pt")
        torch.cuda.empty_cache()


def process(train_path, valid_path, log_dir, architecture):
    """

    Parameters
    ----------
    train_path: contains the path of training values
    valid_path: contains the path of validation values
    log_dir: contains the path where log summary is stored

    Returns
    -------

    """
    parameters = {"batch_size": 32, "shuffle": True, "num_workers": 6}

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    training_set = SrDataset(train_path)
    training_generator = torch.utils.data.DataLoader(training_set, **parameters)

    validation_set = SrDataset(valid_path)
    validation_generator = torch.utils.data.DataLoader(validation_set, **parameters)
    training(training_generator, validation_generator, device, log_dir, architecture)


if __name__ == "__main__":
    arguments = docopt(__doc__)
    train_path = Path(arguments["--train"])
    valid_path = Path(arguments["--valid"])
    log_dir = Path(arguments["--log_dir"])
    architecture = arguments["--architecture"]
    process(train_path, valid_path, log_dir, architecture)
