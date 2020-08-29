from pathlib import Path
import os

import torch
from torchsummary import summary
import torch.optim as optim

from dataset import SrDataset
from docopt import docopt

import numpy as np
from unet import UNET
from losses import SSIM
from Logger import Logger

"""Usage: model.py
model.py --Training_X=X_Train --Training_Y=Y_Train --Valid_X=X_Valid --Valid_Y=Y_Valid

--Training-X=X_Train path  Some directory [default: ./idata]
--Valid-X=X_Valid path  Some directory [default: ./mdata]
--Log-dir=log_dir path Some directory [default: ./kdata]

Example: python3.8 sr/cutter.py --Training_X=idata --Valid_X=mdata --Log_dir=kdata
"""
def normalize_denormalize(norm_denorm_option, tensor_variable, mean, sigma):
    if norm_denorm_option=="norm":
        for i in range(len(tensor_variable)):
            tensor_variable[i] = (tensor_variable[i] - mean[i])/sigma[i]
    else:
        for i in range(len(tensor_variable)):
            tensor_variable[i] = (tensor_variable[i]*sigma[i])+mean[i]
    return tensor_variable

def log_loss_summary(logger, loss, step, prefix=""):
    logger.scalar_summary(prefix + "loss", np.mean(loss), step)

def training(training_generator, validation_generator, device, log_dir):
    '''

    Parameters
    ----------
    training_generator: contains training data
    validation_generator: contains validation data
    device: Cuda Device

    Returns
    -------

    '''
    # parameters
    unet = UNET(inchannels=1, outchannels=1)
    unet.to(device)
    summary(unet, (1, 256, 256), batch_size=-1, device='cuda')
    max_epochs = 50
    criterion = SSIM()
    logger = Logger(str(log_dir))
    step = 0
    for epoch in range(max_epochs):
        unet.train()
        loss_train_list = []
        step += 1
        for i, data in enumerate(training_generator):
            x_train = data['lr']
            y_train = data['hr']
            stat = data['stats']
            mean, sigma = stat['mean'], stat['std']
            x_train, y_train, mean, sigma = x_train.to(device), y_train.to(device), mean.to(device), sigma.to(device)
            x_train = normalize_denormalize("norm", x_train, mean, sigma)
            optimizer = optim.Adam(unet.parameters(), lr=0.0005)
            optimizer.zero_grad()
            with torch.autograd.set_detect_anomaly(True):
                with torch.set_grad_enabled(True):
                    y_pred = unet(x_train)
                    y_pred = normalize_denormalize("denorm", y_pred, mean, sigma)
                    loss_train = criterion(y_pred, y_train)
                    loss_train_list.append(loss_train.item())
                    print("the training loss is {} in epoch {}".format(loss_train.item(), epoch))
                    loss_train.backward()
                    optimizer.step()

            #training log summary after every 10 epochs
            if step%10 == 0:
                log_loss_summary(logger, loss_train_list, step, prefix="train_")
                loss_train_list = []

        del x_train, y_train, mean, sigma, loss_train_list
        loss_valid_list = []
        for i, data in enumerate(validation_generator):
            unet.eval()
            x_valid = data['lr']
            y_valid = data['hr']

            x_valid, y_valid = x_valid.to(device), y_valid.to(device)
            y_pred = unet(x_valid)
            loss_valid = criterion(y_pred, y_valid)
            loss_valid_list.append(loss_valid)
            print("the validation loss is {} in epoch {}".format(loss_valid.item(), epoch))

            #valid log summary after every 10 epochs
            if step % 10 == 0:
                log_loss_summary(logger, loss_valid_list, step, prefix="val_")
                loss_valid_list = []

        del x_valid, y_valid, mean, sigma, loss_valid_list
    torch.save(unet.state_dict(), os.getcwd()+"unet_model.pt")


def process(train_path, valid_path, log_dir):
    '''

    Parameters
    ----------
    train_path: contains the path of training values
    valid_path: contains the path of validation values

    Returns
    -------

    '''
    parameters = {'batch_size': 8,
                  'shuffle': True,
                  'num_workers': 6}

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    training_set = SrDataset(train_path)
    training_generator = torch.utils.data.DataLoader(training_set, **parameters)

    validation_set = SrDataset(valid_path)
    validation_generator = torch.utils.data.DataLoader(validation_set)
    training(training_generator, validation_generator, device, log_dir)




def main():
    arguments = docopt(__doc__, version="Div2k_test")
    train_path = Path(arguments["--Training-X"])
    valid_path = Path(["--Valid-X"])
    log_dir = Path(["--Log-dir"])

    process(train_path, valid_path, log_dir)






