import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
import os
import shutil
from time import time
import datetime
import torch
import torch.nn
from torchsummary import summary
import numpy as np
from vgg_dataset import VGGTrainDataset
from logger import Logger
from train_util import model_selection, check_load_model, model_save, vgg_train, vgg_valid, check_load_pretrained_model, debug_pics

BATCH_SIZE = {
    "vgg" :4
}
LR = {
    "vgg":0.0001
}


def create_dataset(path, lognorm=False, test=False):
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
        print("running PairedDataset")
        return VGGTrainDataset(path, lognorm=lognorm, test=test)
    else:
        raise NotADirectoryError("LR and HR are missing")

def log_loss_summary(logger, loss, step, prefix=""):
    logger.scalar_summary(prefix + "loss", np.mean(loss), step)
def model_draw(logger, model, input_to_model):
    logger.model_graph(model, input_to_model)

def vgg_trainer(training_generator,
    validation_generator,
    device,
    log_dir,
    architecture,
    num_epochs,
    aspp,
    dilation,
    act,
    model_save_path):
    """

    Parameters
    ----------
    training_generator: contains training data
    validation_generator: contains validation data
    device: Cuda Device
    log_dir: The log directory for storing logs
    architecture: The architecture to be used unet or axial
    num_epochs:   The number of epochs
    aspp: True if EDSR is going to use aspp
    dilation True if EDSR is going to use dilation
    act: activation function to be used
    model_save_path: location where model will be saved
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
    summary(model, (1, 256, 256), batch_size=1, device="cuda")
    max_epochs = num_epochs

    '''
    # drawing model
    dummy_input = torch.from_numpy(np.random.randn(1, 1, 64, 64)).float()
    dummy_input = dummy_input.to(device)
    model_draw(logger, model, dummy_input)
    del dummy_input
    '''

    # loading the model
    training_parameters = check_load_model(save_model_path, model, lr)
    training_parameters["criterion"] = torch.nn.CrossEntropyLoss()
    training_parameters["device"] = device
    training_parameters["max_epochs"] = max_epochs
    training_parameters["learning_rate"] = lr
    best_valid_loss = float("inf")
    step = training_parameters["current_epoch"]

    while step < max_epochs:
        print(f"current step is {step}")
        start_time = time()
        model.train()
        # training
        l1_loss, l1_list = vgg_train(training_generator, training_parameters)
        # training log summary
        log_loss_summary(logger, l1_list, step, prefix="_train_l1")
        del l1_list

        # validation
        val_l1_loss, val_l1_list = vgg_valid(validation_generator, training_parameters)
        log_loss_summary(logger, val_l1_list, step, prefix="_val_l1")
        del val_l1_list

        memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        print(
            "\nEpoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} in {:.1f} seconds. [lr:{:.8f}][max mem:{:.0f}MB]".format(
                training_parameters["current_epoch"],
                l1_loss,
                val_l1_loss,
                time() - start_time,
                training_parameters["optimizer"].param_groups[0]["lr"],
                memory,
            ))
        step += 1
        save_params = {
            "epoch": step,
            "model": model.state_dict(),
            "optimizer": training_parameters["optimizer"].state_dict(),
            "scheduler": training_parameters["scheduler"].state_dict(),
        }
        # Save best validation epoch model
        if val_l1_loss < best_valid_loss:
            best_valid_loss = val_l1_loss

            model_save(save_params, f"{str(best_save_model_path)}/{timestamp}_best_model_{step}.pt")
            # deleting the old_best_model after new one is saved

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
        training_parameters["current_epoch"] = step

def vgg_process(
    train_path,
    valid_path,
    log_dir,
    architecture,
    num_epochs,
    lognorm,
    aspp,
    dilation,
    act,
    model_save_path):
    """

    Parameters
    ----------
    train_path
    valid_path
    device
    log_dir
    architecture
    num_epochs
    aspp
    dilation
    act
    model_save_path

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
    validation_set = create_dataset(valid_path, lognorm=lognorm, test=True)
    training_generator = torch.utils.data.DataLoader(training_set, **parameters)
    validation_generator = torch.utils.data.DataLoader(validation_set, **parameters)

    vgg_trainer(
        training_generator,
        validation_generator,
        device,
        log_dir,
        architecture,
        num_epochs,
        aspp,
        dilation,
        act,
        model_save_path
    )