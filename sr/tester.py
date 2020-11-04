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
from unet import UNET
from axial_bicubic import AxialNet
from edsr import EDSR
from tqdm import tqdm
import scipy.ndimage

from PIL import Image, ImageFont, ImageDraw

def writetext(imgfile, e_sr=None, e_lr=None):
    img = Image.open(imgfile)
    width, height = img.size
    draw = ImageDraw.Draw(img)
    font_path = os.path.realpath('./font/dancing.ttf')
    font = ImageFont.truetype(font_path, 15)
    if e_sr is None:
         draw.text((width/2, 0), "LR", font=font, fill=(0, 0, 255))
         draw.text((0, 0), "SR", font=font, fill=(0, 0, 255))
    else:
        draw.text((width/2, 0), "HR", font=font, fill=(0, 0, 255))
        draw.text((0, width/2), "Error", font=font, fill=(0, 0, 255))

        draw.text((0, 0), "SR", font=font)
        draw.text((32, 0), "L1="+str(e_sr[0]), font=font, fill=(255, 0, 0))
        draw.text((32, 16), "PSNR="+str(e_sr[1]), font=font, fill=(255, 0, 0))
        draw.text((32, 32), "SSIM="+str(e_sr[2]), font=font, fill=(255, 0, 0))

        draw.text((height/2, width/2), "LR", font=font, fill=(255, 255, 255))
        draw.text((height/2 + 32, width/2), "L1="+str(e_lr[0]),
                  font=font, fill=(255, 0, 0))
        draw.text((height/2 + 32, width/2 + 16), "PSNR="+str(e_lr[1]),
                  font=font, fill=(255, 0, 0))
        draw.text((height/2 + 32, width/2 + 32), "SSIM="+str(e_lr[2]),
                  font=font, fill=(255, 0, 0))
    img.save("/tmp/test.png")
    os.system("mv /tmp/test.png " + imgfile)

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
    if args["--architecture"] == 'unet':
        model = UNET(in_channels=1, out_channels=1, init_features=32)
    elif args["--architecture"] == 'axial':
        model = AxialNet(num_channels=1, resblocks=2, skip=1)
    elif  args["--architecture"] == 'edsr_16_64':
        model = EDSR(n_resblocks=16, n_feats=64, scale=4,
                     dilation=dilation, aspp=aspp, act=act)
    elif  args["--architecture"] == 'edsr_8_256':
        model = EDSR(n_resblocks=8, n_feats=256, scale=4,
                     dilation=dilation, aspp=aspp, act=act)
    elif  args["--architecture"] == 'edsr_16_256':
        model = EDSR(n_resblocks=16, n_feats=256, scale=4,
                     dilation=dilation, aspp=aspp, act=act)
    elif args["--architecture"] == 'edsr_32_256':
        model = EDSR(n_resblocks=32, n_feats=256, scale=4)

    model.to(device)
    test_loss = 0
    messed_images = 0
    test_psnr = 0.0
    test_ssim = 0.0
    lr_tot_psnr = 0.0
    lr_tot_ssim = 0.0
    model.load_state_dict(torch.load(args["--model"]))
    active_list = []
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
            y_pred =  np.clip(y_pred, stat_test["min"].numpy(), stat_test["max"].numpy())
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
            #x_test = x_test.cpu().numpy()
            #x_test = (x_test * stat_test["std"]) + stat_test["mean"]
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
                plt.imsave(filename, save_plots, cmap="gray", vmin = stat_test["min"], vmax = m+3*s )

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

if __name__ == "__main__":
    args = docopt(__doc__)
    for a in ["--input", "--output", "--model"]:
        args[a] = Path(args[a]).resolve()
    print("Input being read from:", args["--input"])
    #print(args)
    evaluate(args)
    """ An example:
     ./tester hr --input input.npz --output output.npz --model srunet.pt
    {'--help': False,
     '--input': True,
     '--model': True,
     '--output': True,
     '-e': False,
     '-l': False,
     '-p': False,
     'input': 'input_dir',
     'model': 'srunet.pt',
     'output': 'output_dir',
     'architecture': 'arch',
     'hr': True,
     'lr': False}
    """
