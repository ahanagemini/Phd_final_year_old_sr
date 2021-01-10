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

        image = Normalize()(image, stats)
        sample = {
            "lr": image,
            "stats": stats,
        }
        transforms = Compose([Pad(), Reshape(), ToFloatTensor()])
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

    def __call__(self, hr_image, stats):
        """

        Parameters
        ----------
        hr_image: high resolution image
        stats: containing mean and standard deviation

        Returns
        -------
        hr_image: returns normalized hr image
        """
        return (hr_image - stats["mean"]) / stats["std"]


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
        if height > width:
            # start padding width
            diff = height - width
            if diff%2==0:
                left, right = diff//2, diff//2
                x = np.pad(x, [(0, 0), (diff//2, diff//2)])
            else:
                left, right = diff // 2 + 1, diff // 2
                x = np.pad(x, [(0, 0), (diff//2 + 1, diff//2)])
            return x, "width", left, right

        elif width > height:
            diff = width - height
            if diff%2==0:
                top, bottom = diff//2, diff//2
                x = np.pad(x, [(diff//2, diff//2), (0, 0)])
            else:
                top, bottom = diff//2+1, diff//2
                x = np.pad(x, [(diff // 2 + 1, diff // 2), (0, 0)])
            return x, "height", top, bottom

        else:
            return x, "none", 0, 0
    
    def __call__(self, sample):
        """
        
        Parameters
        ----------
        sample

        Returns
        -------

        """
        sample["lr"], sample["type"], sample["pad_1"], sample["pad_2"] = self.pad(sample["lr"])
        print(sample["lr"].shape)
        return sample