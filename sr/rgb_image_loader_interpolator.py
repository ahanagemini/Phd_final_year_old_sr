import torch
import torch.nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

class RGB_Interpolator:
    """This class takes and rgb image and interpolates it and saves it"""
    def t_interpolate(self, image, mode, scale_factor):
        """

        Parameters
        ----------
        image
        mode

        Returns
        -------

        """
        width, height, channels = image.shape
        image = image.reshape((1, channels, height, width))
        image = torch.tensor(image, dtype=torch.float32)
        if mode == "nearest":
            image = F.interpolate(
                image,
                scale_factor=scale_factor,
                mode=mode,
                recompute_scale_factor=False,
            )
        else:
            image = F.interpolate(
                image,
                scale_factor=scale_factor,
                mode=mode,
                align_corners=True,
                recompute_scale_factor=False,
            )

        image = image.numpy()
        image = image[0, :, :, :]
        channels, height, width = image.shape
        image = image.reshape((width, height, channels))
        return image

    def loader(self, ifile):
        image = Image.open(ifile)
        image = image.convert("RGB")
        image = np.array(image)
        return image

    def interpolater_downsampler(self, conf):
        idir = Path(conf.input_dir)
        odir = Path(conf.output_dir)
        if not os.path.isdir(odir):
            os.makedirs(odir)
        image_paths = idir.rglob("*.png")
        for i, image_path in enumerate(tqdm(image_paths)):
            image_name = os.path.splitext(image_path.name)[0]
            image = self.loader(image_path)
            print(image.shape)
            image = self.t_interpolate(image, conf.mode, conf.scale_factor)
            image = Image.fromarray(np.uint8(image))
            image.save(str(odir/(image_name + r".png")))

class Configurator:
    """This is the configurator class"""
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--input_dir", default=os.path.dirname(os.path.abspath(__file__)+r"/input_directory"),
                                 help="This command is used to set input_dir")
        self.parser.add_argument("--output_dir", default=os.path.dirname(os.path.abspath(__file__))+r"/output_directory",
                                 help="This command is used to set output_dir")
        self.parser.add_argument("--mode", default="bicubic",
                                 help="this command is used to set interpolation mode default is bicubic")
        self.parser.add_argument("--scale_factor", type=float, default=0.25,
                                 help="This command is used to set the scaling amount default is 0.25(4X)")

    def parse(self, args=None):
        """Parse the configuration"""
        self.conf = self.parser.parse_args(args=args)
        return self.conf


if __name__=="__main__":
    if (
        len(sys.argv) == 1
        or "--input_dir" not in str(sys.argv)
    ):
        sys.argv.append("-h")
    conf = Configurator().parse()
    colormap = RGB_Interpolator()
    colormap.interpolater_downsampler(conf)
