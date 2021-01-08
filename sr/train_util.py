from pathlib import Path
import os
import scipy.ndimage
import shutil
import torch
import torch.optim as optim
import torch.nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
from tqdm import tqdm
from models import UNET
from models import EDSR
from axial_bicubic import AxialNet
from PIL import Image, ImageFont, ImageDraw
from skimage import metrics

def matrix_cutter(img, width=64, height=64):
    """


    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    height : TYPE, optional
        DESCRIPTION. The default is 256.
    width : TYPE, optional
        DESCRIPTION. The default is 256.

    Returns
    -------
    None.

    """
    images = []
    img_batch, img_channels, img_height, img_width = img.size()

    # check if images have 256 width and 256 height if it does skip cutting
    if img_height <= height and img_width <= width:
        return [(0, 0, img)]

    for i, ih in enumerate(range(0, img_height, height)):
        for j, iw in enumerate(range(0, img_width, width)):
            posx = iw
            posy = ih
            if posx + width > img_width:
                posx = img_width - width
            if posy + height > img_height:
                posy = img_height - height

            cutimg = img[:, :, posy : posy + height, posx : posx + width]
            cutimg_batch, cutimg_channels, cutimg_height, cutimg_width = cutimg.size()
            assert cutimg_height == height and cutimg_width == width
            images.append((i, j, cutimg))
    return images

def matrix_stitcher(img, mat_dict, device, scale=4, height=256, width=256):
    img_batch, img_channel, img_height, img_width = img.size()
    del img
    img_height, img_width = img_height*scale, img_width*scale

    img = torch.zeros((img_batch, img_channel, img_height, img_width), device=device)
    for i, ih in enumerate(range(0, img_height, height)):
        for j, iw in enumerate(range(0, img_width, width)):
            posx = iw
            posy = ih
            if posx + width > img_width:
                posx = img_width - width
            if posy + height > img_height:
                posy = img_height - height
            img[:, :, posy:posy + height, posx:posx + height] = mat_dict[str(i)+str(j)]

    return img


def chop_forward(x, model, device):
    images = matrix_cutter(x)
    up_images = {}
    for i, j, mat in images:
        up_images[str(i)+str(j)] = model(mat)
    out_image = matrix_stitcher(x, up_images, device)
    return out_image


def writetext(imgfile, e_sr=None, e_lr=None):
    img = Image.open(imgfile)
    width, height = img.size
    draw = ImageDraw.Draw(img)
    font_path = os.path.abspath(os.path.expanduser('./font/dancing.ttf'))
    font = ImageFont.truetype(font_path, 15)
    if e_sr is None:
        draw.text((width / 2, 0), "LR", font=font, fill=(0, 0, 255))
        draw.text((0, 0), "SR", font=font, fill=(0, 0, 255))
    else:
        draw.text((width / 2, 0), "HR", font=font, fill=(0, 0, 255))
        draw.text((0, width / 2), "Error", font=font, fill=(0, 0, 255))

        draw.text((0, 0), "SR", font=font)
        draw.text((32, 0), "L1=" + str(e_sr[0]), font=font, fill=(255, 0, 0))
        draw.text((32, 16), "PSNR=" + str(e_sr[1]), font=font, fill=(255, 0, 0))
        draw.text((32, 32), "SSIM=" + str(e_sr[2]), font=font, fill=(255, 0, 0))

        draw.text((height / 2, width / 2), "LR", font=font, fill=(255, 255, 255))
        draw.text((height / 2 + 32, width / 2), "L1=" + str(e_lr[0]),
                  font=font, fill=(255, 0, 0))
        draw.text((height / 2 + 32, width / 2 + 16), "PSNR=" + str(e_lr[1]),
                  font=font, fill=(255, 0, 0))
        draw.text((height / 2 + 32, width / 2 + 32), "SSIM=" + str(e_lr[2]),
                  font=font, fill=(255, 0, 0))
    img.save("/tmp/test.png")
    os.system("mv /tmp/test.png " + imgfile)


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
        model = EDSR(n_resblocks=32, n_feats=64, aspp=aspp, dilation=dilation, act=act)
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
    training_parameters = {}

    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.999, verbose=True)

    if not current_model_list:
        current_model = ""
    else:
        current_model = current_model_list[-1]

    current_epoch = 0
    # check if there is a .pt file of model after an epoch
    training_parameters["model"] = model
    training_parameters["optimizer"] = optimizer
    training_parameters["scheduler"] = scheduler
    training_parameters["current_epoch"] = current_epoch
    training_parameters["current_model"] = current_model
    if os.path.isfile(current_model):
        load_model(current_model, training_parameters)
    for param_group in training_parameters["optimizer"].param_groups:
        param_group['lr'] = learning_rate
    training_parameters["scheduler"].factor = 0.999
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
    model_list = list(model_folder.rglob("*.pt"))
    if not model_list:
        model_current = ""
    else:
        model_current = model_list[-1]


    if not model_folder.is_dir():
        os.makedirs(model_folder)
    torch.save(save_params, model_path)

    # deleting the old model after generating new one
    if os.path.isfile(model_current):
        os.remove(model_current)

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
    device = training_parameters["device"]
    criterion = training_parameters["criterion"]
    row_diff_loss = training_parameters["row_diff_loss"]
    lambda_row = training_parameters["lambda_row"]
    loss_train_list = []
    l1_loss_list = []
    row_diff_list = []
    train_loss = row_loss = l1_loss = 0.0
    model.train(True)
    for batch_idx, data in enumerate(tqdm(training_generator)):
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
        training_parameters['optimizer'].zero_grad()
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
                training_parameters['optimizer'].step()


    return loss_train_list, train_loss, l1_loss, row_loss, l1_loss_list, row_diff_list


def valid_model(validation_generator, training_parameters):
    """

    Parameters
    ----------
    validation_generator: The validation dataloader
    training_parameters: It contains all the parameters required for training

    Returns
    -------
    """
    model = training_parameters["model"]
    device = training_parameters["device"]
    criterion = training_parameters["criterion"]
    row_diff_loss = training_parameters["row_diff_loss"]
    lambda_row = training_parameters["lambda_row"]
    loss_valid_list = []
    l1_loss_valid_list = []
    row_diff_valid_list = []
    valid_loss = valid_row_loss = valid_l1_loss = 0.0
    model.train(False)
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(validation_generator)):

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
        training_parameters['scheduler'].step(valid_loss)

    return loss_valid_list, valid_loss, valid_l1_loss, valid_row_loss, l1_loss_valid_list, row_diff_valid_list


def test_model(test_generator, testing_parameters):
    """

    Parameters
    ----------
    test_generator: The testing dataloader
    testing_parameters: It contains all the parameters required for testing

    Returns
    -------
    """
    model = testing_parameters["model"]
    args = testing_parameters["args"]
    lognorm = testing_parameters["lognorm"]
    device = testing_parameters["device"]
    active_learning = testing_parameters["active"]
    active_list = []
    test_loss = 0
    messed_images = 0
    test_psnr = 0.0
    test_ssim = 0.0
    lr_tot_psnr = 0.0
    lr_tot_ssim = 0.0
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_generator)):
            model.eval()
            # unet.train(False)
            x_test = data["lr"]
            y_test = data["hr"]
            stat_test = data["stats"]
            x_test = x_test.to(device)
            y_test = y_test.numpy()
            y_pred = model(x_test)

            # Inverse of the nomalizations used. Applied on SR data
            stat_test["std"] = stat_test["std"].numpy()
            stat_test["mean"] = stat_test["mean"].numpy()
            y_pred = y_pred.cpu().numpy()
            y_pred = (y_pred * stat_test["std"]) + stat_test["mean"]
            y_pred = np.clip(y_pred, stat_test["min"].numpy(), stat_test["max"].numpy())
            if lognorm:
                image_sign = np.sign(y_pred)
                y_pred = image_sign * (np.exp(np.abs(y_pred)) - 1.0)

            # Compute the loss if HR is given
            if args["hr"]:
                loss_test = np.mean(np.abs(y_pred - y_test))
                # loss_test = loss_test / stat_test["std"]
                if not active_learning:
                    sr_hr = np.hstack([y_pred.reshape(-1, y_pred.shape[-1]),
                                       y_test.reshape(-1, y_pred.shape[-1])])

            # Inverse normalize the LR image
            # x_test = x_test.cpu().numpy()
            # x_test = (x_test * stat_test["std"]) + stat_test["mean"]
            x_test = data["lr_un"].cpu().numpy()
            if lognorm:
                image_sign = np.sign(x_test)
                x_test = image_sign * (np.exp(np.abs(x_test)) - 1.0)

            if args["kernel"]:
                x_test = x_test.reshape(-1, x_test.shape[-1])
                x_test = scipy.ndimage.zoom(x_test, 4.0)

            # Create and save image consisting of error map, HR, SR, LR
            if not active_learning:
                if args["hr"]:
                    error = np.abs(y_pred - y_test).reshape(-1, y_pred.shape[-1])
                    error = error * (stat_test["max"].numpy() / np.max(error))
                    lr_error = np.hstack([error, x_test])
                    save_plots = np.vstack([sr_hr, lr_error])
                else:
                    save_plots = np.hstack(
                        [y_pred.reshape(-1, y_pred.shape[-1]),
                         x_test.reshape(-1, y_pred.shape[-1])]
                    )
                if args["--save_slice"]:
                    save_img = y_pred.astype(np.uint16)
                    shutil.rmtree(str(args["--output"]) + "/slices")
                    os.mkdir(str(args["--output"]) + "/slices")
                    outFilepath = str(args["--output"]) + "/slices/" + data["file"][0] + ".tiff"
                    tifffile.imsave(outFilepath, save_img)
                filename = os.path.join(args["--output"], f"{batch_idx}.png")
                m = np.mean(save_plots)
                s = np.std(save_plots)
                plt.imsave(filename, save_plots, cmap="gray", vmin=stat_test["min"], vmax=m + 3 * s)

            if not args["hr"]:
                writetext(os.path.abspath(filename))
            if args["hr"]:
                test_loss = test_loss + loss_test
                lr_l1 = np.mean(np.abs(y_test - x_test))
                sr_l1 = np.mean(np.abs(y_test - y_pred))
                x_test = x_test.reshape(x_test.shape[1], -1)
                y_test = y_test.reshape(y_test.shape[3], -1)
                y_pred = y_pred.reshape(y_pred.shape[3], -1)
                data_range = stat_test["max"].numpy()
                lr_psnr = metrics.peak_signal_noise_ratio(y_test, x_test,
                                                          data_range=data_range)
                sr_psnr = metrics.peak_signal_noise_ratio(y_test, y_pred,
                                                          data_range=data_range)
                lr_ssim = metrics.structural_similarity(y_test, x_test,
                                                        gaussian_weights=True,
                                                        data_range=data_range)
                sr_ssim = metrics.structural_similarity(y_test, y_pred,
                                                        gaussian_weights=True,
                                                        data_range=data_range)
                test_psnr = test_psnr + sr_psnr[0]
                test_ssim = test_ssim + sr_ssim
                lr_tot_psnr = lr_tot_psnr + lr_psnr[0]
                lr_tot_ssim = lr_tot_ssim + lr_ssim
                srerror = (sr_l1, sr_psnr[0], sr_ssim)
                lrerror = (lr_l1, lr_psnr[0], lr_ssim)
                if not active_learning:
                    writetext(os.path.abspath(filename), srerror, lrerror)
                if sr_l1 > lr_l1:
                    messed_images = messed_images + 1
                if active_learning:
                    active_list.append([data["file"][0], sr_l1, sr_psnr[0],
                                        sr_ssim, lr_l1, lr_psnr[0], lr_ssim])

        if args["hr"]:
            print("\nTest Loss: {:.6f}".format(test_loss / len(test_generator)))
            print("\nTest PSNR: {:.6f}".format(test_psnr / len(test_generator)))
            print("\nTest SSIM: {:.6f}".format(test_ssim / len(test_generator)))
            print("\nLR PSNR: {:.6f}".format(lr_tot_psnr / len(test_generator)))
            print("\nLR SSIM: {:.6f}".format(lr_tot_ssim / len(test_generator)))
            print(f"NUmber of images where SR L1 is worse than LR: {messed_images}")
            if active_learning:
                df = pd.DataFrame(active_list, columns=['filename', 'SR_L1', 'SR_PSNR',
                                                        'SR_SSIM', 'LR_L1', 'LR_PSNR', 'LR_SSIM'])
                active_file = str(args["--input"]).split('/')[-1]
                df.to_csv(f"active_metrics/{active_file}.csv")
