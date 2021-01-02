from pathlib import Path
import os
import scipy.ndimage

import torch
import torch.optim as optim
import torch.nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from models import UNET
from models import EDSR
from axial_bicubic import AxialNet

def model_selection(architecture, aspp, dilation, act):
    """

    Parameters
    ----------
    architecture
    aspp
    dilation
    act
    Returns
    -------

    """
    if architecture == "unet":
        model = UNET(in_channels=1, out_channels=1, init_features=64, depth=4)
    elif architecture == "axial":
        model = AxialNet(num_channels=1, resblocks=2, skip=1)
    elif architecture == "edsr_16_64":
        model = EDSR(n_resblocks=32, n_feats=256, aspp=aspp, dilation=dilation, act=act)
    elif architecture == "edsr_8_256":
        model = EDSR(n_resblocks=8, n_feats=256, aspp=aspp, dilation=dilation, act=act)
    elif architecture == "edsr_16_256":
        model = EDSR(n_resblocks=16, n_feats=256, aspp=aspp, dilation=dilation, act=act)
    elif architecture == "edsr_32_256":
        model = EDSR(n_resblocks=32, n_feats=256)
    return model

def debug_pics(data, stat, input_save_path):
    """

    Parameters
    ----------
    x_train
    y_train

    Returns
    -------

    """
    x_train = data["lr"]
    y_train = data["hr"]
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
            x_rescale_pad = scipy.ndimage.zoom(x_rescale_pad, 4.0)
            save_plots = np.hstack(
                [x_rescale_pad, y_rescale.reshape(y_rescale.shape[1], -1)]
            )
            save_plots = np.clip(
                save_plots, stat["min"][i].numpy(), stat["max"][i].numpy()
            )
            vmax = stat["mean"][i].numpy() + 3 * stat["std"][i].numpy()
            vmin = stat["min"][i].numpy()
            filename = os.path.join(f"{input_save_path}/{filename}.tiff")
            plt.imsave(filename, save_plots, vmin=vmin, vmax=vmax, cmap="gray")


def load_model(model_path, training_parameters):
    checkpoint = torch.load(model_path)
    training_parameters["model"].load_state_dict(checkpoint["model"])
    training_parameters["current_epoch"] = checkpoint["epoch"]
    training_parameters["optimizer"].load_state_dict(checkpoint["optimizer"])
    training_parameters["scheduler"].load_state_dict(checkpoint["scheduler"])



def check_load_model(save_model_path, model, learning_rate=0.0005):
    """

    Parameters
    ----------
    save_model_path
    model
    learning_rate
    Returns
    -------

    """
    current_save_model_path = save_model_path / "current"
    best_save_model_path = save_model_path / "best"
    if not current_save_model_path.is_dir():
        os.makedirs(current_save_model_path)
    if not best_save_model_path.is_dir():
        os.makedirs(best_save_model_path)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    current_model_list = list(current_save_model_path.rglob("*.pt"))
    current_model_list = sorted(current_model_list)
    best_model_list = list(best_save_model_path.rglob("*best_model.pt"))
    best_model_list = sorted(best_model_list)
    training_parameters = {}

    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1)

    if not current_model_list:
        current_model = ""
    else:
        current_model = current_model_list[-1]

    if not best_model_list:
        best_model = ""
    else:
        best_model = best_model_list[-1]

    current_epoch = 0
    # check if there is a .pt file of model after an epoch
    training_parameters["model"] = model
    training_parameters["optimizer"] = optimizer
    training_parameters["scheduler"] = scheduler
    training_parameters["current_epoch"] = current_epoch
    training_parameters["best_model"] = best_model
    training_parameters["current_model"] = current_model
    if os.path.isfile(current_model):
        load_model(current_model, training_parameters)

    # check if there is a .pt file of best model if epoch .pt file is missing
    elif os.path.isfile(best_model):
        load_model(best_model, training_parameters)

    return training_parameters


def model_save(save_params, train_model_path):
    """

    Parameters
    ----------
    save_params: dictionary containing all params to be saved
    train_model_path: model path

    Returns
    -------

    """
    model_path = Path(train_model_path)
    model_folder = model_path.parent
    if not model_folder.is_dir():
        os.makedirs(model_folder)
    torch.save(save_params, model_path)


def train_model(training_generator, training_parameters):
    """

    Parameters
    ----------
    training_generator: The training dataloader
    training_parameters: It contains all the parameters required for training

    Returns
    -------

    """
    model = training_parameters["model"]
    optimizer = training_parameters["optimizer"]
    scheduler = training_parameters["scheduler"]
    device = training_parameters["device"]
    criterion = training_parameters["criterion"]
    row_diff_loss = training_parameters["row_diff_loss"]
    lambda_row = training_parameters["lambda_row"]
    lr = training_parameters["learning_rate"]
    loss_train_list = []
    l1_loss_list = []
    row_diff_list = []
    train_loss = row_loss = l1_loss = 0.0
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
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
            with torch.set_grad_enabled(True):
                y_pred = model(x_train)
                loss_l1 = criterion(y_pred, y_train)
                loss_row = lambda_row * row_diff_loss(y_pred, y_train)
                loss_train = loss_l1 + loss_row
                train_loss = train_loss + (
                        (1 / (batch_idx + 1)) * (loss_train.data - train_loss)
                )
                l1_loss = l1_loss + (
                        (1 / (batch_idx + 1)) * (loss_l1.data - l1_loss))

                row_loss = row_loss + (
                        (1 / (batch_idx + 1)) * (loss_row.data - row_loss)
                )
                loss_train_list.append(loss_train.item())
                l1_loss_list.append(loss_l1.item())
                row_diff_list.append(loss_row.item())
                loss_train.backward()
                optimizer.step()

    return loss_train_list, train_loss, l1_loss, row_loss, l1_loss_list, row_diff_list

def valid_model(validation_generator, training_parameters):
    model = training_parameters["model"]
    optimizer = training_parameters["optimizer"]
    scheduler = training_parameters["scheduler"]
    device = training_parameters["device"]
    criterion = training_parameters["criterion"]
    row_diff_loss = training_parameters["row_diff_loss"]
    lambda_row = training_parameters["lambda_row"]
    epoch = training_parameters["current_epoch"]
    max_epochs = training_parameters["max_epochs"]
    loss_valid_list = []
    l1_loss_valid_list = []
    row_diff_valid_list = []
    valid_loss = valid_row_loss = valid_l1_loss = 0.0
    scheduler.factor = 1 + (epoch / max_epochs) ** 0.9
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(validation_generator)):
            # unet.eval()
            model.train(False)
            x_valid = data["lr"]
            y_valid = data["hr"]

            x_valid, y_valid = x_valid.to(device), y_valid.to(device)
            y_pred = model(x_valid)
            loss_l1_valid = criterion(y_pred, y_valid)
            loss_row_valid = lambda_row * row_diff_loss(y_pred, y_valid)
            loss_valid = loss_l1_valid + loss_row_valid
            loss_valid_list.append(loss_valid.item())
            l1_loss_valid_list.append(loss_l1_valid.item())
            row_diff_valid_list.append(loss_row_valid.item())
            valid_loss = valid_loss + (
                    (1 / (batch_idx + 1)) * (loss_valid.data - valid_loss)
            )
            valid_row_loss = valid_row_loss + (
                    (1 / (batch_idx + 1)) * (loss_row_valid.data - valid_row_loss))
            valid_l1_loss = valid_l1_loss + (
                    (1 / (batch_idx + 1)) * (loss_l1_valid.data - valid_l1_loss))
        scheduler.step(valid_loss)

    return loss_valid_list, valid_loss, valid_l1_loss, valid_row_loss, l1_loss_valid_list, row_diff_valid_list