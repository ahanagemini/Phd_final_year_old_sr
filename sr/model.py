from pathlib import Path
import os

import torch
from torchsummary import summary
import torch.optim as optim

from docopt import docopt

from unet import UNET
from dataloader import DataSet
from losses import PSNR

"""Usage: model.py
model.py --Training_X=X_Train --Training_Y=Y_Train --Valid_X=X_Valid --Valid_Y=Y_Valid

--Training_X=X_Train path  Some directory [default: ./x_tdata]
--Training_Y=Y_Train path  Some directory [default: ./y_tdata]
--Valid_X=X_Valid path  Some directory [default: ./x_vdata]
--Valid_Y=Y_Valid path  Some directory [default: ./y_vdata]

Example: python3.8 sr/cutter.py --Training_X=x_tdata --Training_Y=y_tdata --Valid_X=x_vdata --Valid_Y=y_vdata
"""

lossTrain = []
def training(training_generator, validation_generator, device):
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
    unet = UNET(inchannels=3, outchannels=3)
    unet.to(device)
    summary(unet.cuda(), (3, 256, 256))
    max_epochs = 100
    for epoch in range(max_epochs):
        unet.train()
        for x_train, y_train in training_generator:
            x_train, y_train = x_train.to(device), y_train.to(device)
            criterion = PSNR()
            optimizer = optim.Adam(unet.parameters(), lr=0.0005)
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase='train'):
                y_pred = unet(x_train)

                loss = criterion(y_pred, y_train)
                print("the training loss is {} in epoch {}".format(loss.item(), epoch))
                loss.backward()
                optimizer.step()
        for x_valid, y_valid in validation_generator:
            unet.eval()
            x_valid, y_valid = x_valid.to(device), y_valid.to(device)
            y_pred = unet(x_valid)

            loss = criterion(y_pred, y_valid)
            print("the validation loss is {} in epoch {}".format(loss.item(), epoch))
    torch.save(unet.state_dict(), os.getcwd())

def process(x_train, y_train, x_valid, y_valid):
    '''

    Parameters
    ----------
    X_train: contains training values
    Y_train: contains training label values
    X_valid: contains validation values
    Y_valid: contains validation label values

    Returns
    -------

    '''
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    parameters = {'batch_size': 64,
                  'shuffle': True,
                  'num_workers': 6}
    training_set = DataSet(x_train, y_train)
    training_generator = torch.utils.data.DataLoader(training_set, **parameters)

    validation_set = DataSet(x_valid, y_valid)
    validation_generator = torch.utils.data.DataLoader(validation_set)
    training(training_generator, validation_generator, device)

def scan_idir(ipath):
    """
    Returns (x,y) pairs so that x can be processed to create y
    """
    extensions = ["*.npy", "*.npz", "*.png", "*.jpg", "*.gif", "*.jpeg"]
    files_list = []
    [files_list.extend(sorted(ipath.rglob(x))) for x in extensions]
    return files_list


def main():
    arguments = docopt(__doc__, version="Div2k_test")
    train_path = Path(arguments["--Training_X"])
    train_label_path = Path(arguments["--Training_Y"])

    valid_path = Path(["--Valid_X"])
    valid_label_path = Path(["--Valid_y"])

    x_train = scan_idir(train_path)
    y_train = scan_idir(train_label_path)
    x_valid = scan_idir(valid_path)
    y_valid = scan_idir(valid_label_path)

    process(x_train, y_train, x_valid, y_valid)






