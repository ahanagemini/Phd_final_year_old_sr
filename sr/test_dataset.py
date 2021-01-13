from pathlib import Path
import json
import random
import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import scipy.ndimage
import numpy as np
from stat_plotter import PlotStat
from cutter import loader

class Upsampler_Dataset(Dataset):
    """This dataloader will read the images for upsampling"""
    def __init__(self, root_dir, lognorm=False):
        """

        Parameters
        ----------
        root_dir
        lognorm
        """
        self.lognorm = lognorm
        self.root_dir = Path(root_dir).expanduser().resolve().absolute()
        self.imagelist = []
        self.statlist = []
        for image_file in self.root_dir.glob("*.npz"):
            image_parent = image_file.parent
            self.imagelist.append(image_file)
            self.statlist.append(image_parent / "stats.json")

        print(f"Number of images found {len(self.imagelist)}")

    def __len__(self):
        return len(self.imagelist)

    def __getitem__(self, idx):
        image_file = self.imagelist[idx]
        keys = ("mean", "std", "min", "max")
        stat_file = self.statlist[idx]
        image = loader(image_file)
        stats = json.load(open(stat_file))
        stats = {x: stats[x] for x in keys}

        sample = {
            "lr": image,
            "stats": stats,
        }
        transforms = Compose([Pad(), Normalize(), Reshape(), ToFloatTensor()])
        for i, trans in enumerate([transforms]):
            sample = trans(sample)
        return sample


class Interpol_Test_Dataset(Dataset):
    """This dataset is for evaluating models using different interpolations"""
    def __init__(self, root_dir, lognorm=False, interpolation_type="bicubic"):
        self.interpolation_type=interpolation_type
        self.lognorm = lognorm
        self.root_dir = Path(root_dir).expanduser().resolve().absolute()
        self.imagelist = []
        self.statlist = []
        for image_file in self.root_dir.glob("*.npz"):
            image_parent = image_file.parent
            self.imagelist.append(image_file)
            self.statlist.append(image_parent / "stats.json")

        print(f"Number of images found {len(self.imagelist)}")

    def __len__(self):
        return len(self.imagelist)

    def __getitem__(self, idx):
        image_file = self.imagelist[idx]
        keys = ("mean", "std", "min", "max")
        stat_file = self.statlist[idx]
        image = loader(image_file)
        image_unchanged = image.copy()
        stats = json.load(open(stat_file))
        stats = {x: stats[x] for x in keys}
        sample = {
            "lr": image,
            "lr_un": image,
            "stats": stats,
        }
        transforms = Compose([Interpolate(self.interpolation_type), Pad(), Normalize(), Reshape(), ToFloatTensor()])
        for i, trans in enumerate([transforms]):
            sample = trans(sample)
        return sample


class Reshape:
    """Reshaping tensors"""

    def __call__(self, sample):
        """

        Parameters
        ----------
        sample: dictionary containing lr, hr and stats

        Returns
        -------
        sample: dictionary containing reshaped lr and reshaped hr
        """

        # setting lr width and height
        lr_height, lr_width = sample["lr"].shape

        sample["lr"] = np.reshape(sample["lr"], (1, lr_height, lr_width))
        return sample


class Normalize:
    """Normalizing the high resolution image using mean and standard deviation"""

    def __call__(self, sample):
        """

        Parameters
        ----------
        hr_image: high resolution image
        stats: containing mean and standard deviation

        Returns
        -------
        hr_image: returns normalized hr image
        """
        sample["lr"] = (sample["lr"] - sample["stats"]["mean"]) / sample["stats"]["std"]
        return sample


class ToFloatTensor:
    """This class is for converting the image array to Float Tensor"""

    def __call__(self, sample):
        """
        Parameters
        ----------
        sample: dictionary containing lr, hr and stats

        Returns
        -------
        sample: dictionary containing transformed lr and transformed hr
        """
        sample["lr"] = torch.from_numpy(sample["lr"].copy()).float()

        return sample
    
class Pad:
    """This class will pad the image to equal size. Will only work for 2d images"""

    def power_of_2_next(self, x):
        p = 1
        if (x and not (x & (x - 1))):
            return x
        while (p < x):
            p <<= 1
        return p
    
    def pad(self, x):
        height, width = x.shape

        # calculating the nearest power of 2 of the max value between height and width
        # the image is then padded and the top bottom left and
        max_val = max(height, width)
        max_val = self.power_of_2_next(max_val)
        diff_height, diff_width = max_val - height, max_val-width
        if diff_height % 2 == 0:
            top = bottom = diff_height // 2
        else:
            top = diff_height // 2 + 1
            bottom = diff_height // 2

        if diff_width % 2 == 0:
            left = right = diff_width // 2
        else:
            left = diff_width // 2 + 1
            right = diff_width // 2

        x = np.pad(x, ((top, bottom), (left, right)))
        return x, top, bottom, left, right
    
    def __call__(self, sample):
        """
        
        Parameters
        ----------
        sample

        Returns
        -------

        """
        sample["lr"], sample["top"], sample["bottom"], sample["left"], sample["right"] = self.pad(sample["lr"])
        return sample

class Interpolate:
    """This class will interpolate the images"""
    def __init__(self, interpolation_type):
        self.interpolation_type = interpolation_type

    def __call__(self, sample):
        interpol_type = PlotStat()
        if self.interpolation_type == "scipy":
            sample["lr"] = interpol_type.scipy_zoom(interpol_type.scipy_zoom(sample["lr"], 0.25), 0.25)
        elif self.interpolation_type == "pil_image_resize":
            sample["lr"] = interpol_type.pil_image(interpol_type.pil_image(sample["lr"], 0.25), 0.25)
        else:
            sample["lr"] = interpol_type.t_interpolate(interpol_type.t_interpolate(sample["lr"], mode=self.interpolation_type,
                                                                            scale_factor=0.25),
                                                mode=self.interpolation_type, scale_factor=0.25)
        return sample