import os
import sys
import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from test_dataset import Interpol_Test_Dataset
from train_util import model_selection, forward_chop
from PIL import Image, ImageDraw, ImageFont
from skimage import metrics


def writetext(imgfile, l1, psnr, ssim):
    img = Image.open(imgfile)
    draw = ImageDraw.Draw(img)
    font_path = os.path.abspath(os.path.expanduser("./font/dancing.ttf"))
    font = ImageFont.truetype(font_path, 10)
    draw.text((0, 0), "L1=" + str(l1), font=font, fill=(255, 0, 0))
    draw.text((0, 16), "PSNR=" + str(psnr), font=font, fill=(255, 0, 0))
    draw.text((0, 32), "SSIM=" + str(ssim), font=font, fill=(255, 0, 0))
    img.save(imgfile)
def prepare(tensor, conf):
    """

    Parameters
    ----------
    tensor
    conf

    Returns
    -------

    """
    if conf.precision == 'half': tensor = tensor.half()
    return tensor

def create_dataset(path, lognorm=False, interpol="bicubic"):
    """

    Parameters
    ----------
    path: path to data directory
    lognorm: Is log normalization used?
    interpol: type of interpolation(bicubic, bilinear, scipy, pil_resize)

    Returns
    -------
    Loaded dataset

    """
    return Interpol_Test_Dataset(path, lognorm, interpolation_type=interpol)

class ModelTester:
    """This class will contain methods that will load the model and images from input directory, downsample the images
    teice using the given interpolations and then upsample two times the images and then calculate the metrics like l1
    ssim and psnr between the ground truth and the sr image and will save the image with the following values"""
    def upsampler(self, conf):
        parameters = {"batch_size": 1, "shuffle": False, "num_workers": 6}
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        idir = Path(conf.input)
        if not os.path.isdir(Path(conf.output)):
            os.makedirs(Path(conf.output))

        test_set = create_dataset(idir, conf.lognorm, conf.interpol)
        test_generator = torch.utils.data.DataLoader(test_set, **parameters)
        model = model_selection(conf.architecture, conf.aspp, conf.dilation, conf.act)
        model = model.to(device)
        checkpoint = torch.load(conf.model)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        if conf.precision == "half":
            model.half()
        with torch.no_grad():
            for i, sample in enumerate(tqdm(test_generator)):
                stats = sample["stats"]
                mean = stats["mean"]
                std = stats["std"]
                mean = mean.to(device)
                std = std.to(device)
                sample["lr"] = prepare(sample["lr"], conf)
                sample["lr_un"] = prepare(sample["lr_un"], conf)
                y_pred = forward_chop(forward_chop(sample["lr"].to(device), model=model, shave=16, min_size=16384),
                                      model=model, shave=32, min_size=16384)
                y_pred = (y_pred * std) + mean
                if conf.lognorm:
                    image_sign = np.sign(y_pred)
                    y_pred = image_sign * (np.exp(np.abs(y_pred)) - 1.0)
                    del image_sign
                y = sample["lr_un"].numpy()
                y = y[0, :, :]
                vmax = np.max(y)
                vmin = np.min(y)
                height_y, width_y = y.shape
                y_pred = y_pred.reshape(-1, y_pred.shape[-1])
                y_pred = y_pred.cpu().numpy()
                height, width = y_pred.shape
                y_pred = y_pred[sample["top"]*4: height-sample["bottom"]*4, sample["left"]*4:width-sample["right"]*4]
                filename = str(i)
                filepath = conf.output + fr"/{filename}_test.png"
                plt.imsave(filepath, y_pred, vmax=vmax, vmin=vmin, cmap="gray")
                y_pred = y_pred[:height_y, :width_y]
                data_range = stats["max"].numpy()
                # calculating l1 loss, ssim loss and psnr loss
                l1_loss = np.mean(np.abs(y - y_pred))
                ssim_loss = metrics.structural_similarity(y, y_pred, data_range=data_range)
                psnr_loss = metrics.peak_signal_noise_ratio(y, y_pred, data_range=data_range)
                horizontal_stack = np.hstack((y_pred, y))
                filepath = conf.output + fr"/{filename}.png"
                plt.imsave(filepath, horizontal_stack, vmax=vmax, vmin=vmin, cmap="gray")
                writetext(filepath, l1_loss, psnr_loss, ssim_loss)





class Configurator:
    """ This is the config class for tester"""

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--input', default=os.path.dirname(os.path.abspath(__file__)) + r"/input_dir",
                                 help="use this command to set input directory")
        self.parser.add_argument('--output', default=os.path.dirname(os.path.abspath(__file__)) + r"/output_dir",
                                 help="use this command to set output directory")
        self.parser.add_argument('--model', default=os.path.dirname(os.path.abspath(__file__)) + r"/model_dir",
                                 help="use this command to set model directory")
        self.parser.add_argument('--architecture', default="edsr_16_64",
                                 help="use this command to set architecture")
        self.parser.add_argument('--act', default="leakyrelu",
                                 help="use this command to set activation")
        self.parser.add_argument('--aspp', default=False,
                                 help="use this to set aspp for edsr")
        self.parser.add_argument('--dilation', default=False,
                                 help="use this to set dilation for edsr")
        self.parser.add_argument('--lognorm', default=False,
                                 help="use this command to set lognorm")
        self.parser.add_argument('--precision', default='half',
                                 help="use this to set the command to change the precision")
        self.parser.add_argument("--interpol", default="bicubic",
                                 help="Use this command to set the interpolation (bicubic, bilinear, scipy, pil_resize")

    def parse(self, args=None):
        """Parse the configuration"""
        self.conf = self.parser.parse_args(args=args)
        return self.conf

if __name__ == "__main__":
    conf = Configurator().parse()
    if not os.path.isdir(Path(conf.output)):
        os.makedirs(Path(conf.output))
    modeltester = ModelTester()
    modeltester.upsampler(conf)