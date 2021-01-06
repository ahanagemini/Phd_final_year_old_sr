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
        stat_file = self.statlist[idx]
        image = loader(image_file)
        stats = json.load(open(stat_file))

        image = Normalize()(image, stats)
        sample = {
            "lr": image,
            "stats": stats,
        }
        transforms = Compose([Reshape(), ToFloatTensor()])
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