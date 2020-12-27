"""
Dataset file
"""
from pathlib import Path
import json
import random
import os

from kernelgan import imresize

# from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import scipy.ndimage
import numpy as np
from cutter import loader


class PairedDataset(Dataset):
    """Dataset class for loading large amount of image arrays data"""

    def __init__(
        self, root_dir, lognorm=False, test=False
    ):
        """
        Args:
            root_dir (string): Directory with all the images.
            lognorm: True if we ar eusing log normalization
            test: True only for test dataset
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = Path(root_dir).expanduser().resolve().absolute()
        self.lr_dir = self.root_dir / "LR"
        self.hr_dir = self.root_dir / "HR"
        self.lr_hr_tuple = []
        self.lr_hr_stat_tuple = []
        # Todo : Write a function to find all corresponding images correctly
        # After that randomly shuffle those pairs
        for image_file_name in self.lr_dir.rglob("*.npz"):
            image_name = image_file_name.name
            image_parent_name = os.path.splitext(image_file_name.parent.name)[0]
            lr_image = image_file_name
            hr_image = self.hr_dir / image_parent_name / image_name
            self.lr_hr_tuple.append((lr_image, hr_image))
        np.random.shuffle(self.lr_hr_tuple)
        self.lognorm = lognorm
        self.test = test
        for ftuple in self.lr_hr_tuple:
            lr_image, hr_image = ftuple
            lstat = json.load(open(str(lr_image.parent / "stats.json")))
            hstat = json.load(open(str(hr_image.parent / "stats.json")))
            self.lr_hr_stat_tuple.append((lstat, hstat))

        print("Total number of data elements found = ", len(self.lr_hr_tuple))
        print(f"Total number of stat files found = {len(self.lr_hr_stat_tuple)}")

    def __len__(self):
        return len(self.lr_hr_tuple)

    def __getitem__(self, idx):
        lrimg_name, hrimg_name = self.lr_hr_tuple[idx]
        lrimg_name = Path(lrimg_name)
        hrimg_name = Path(hrimg_name)
        filename = os.path.basename(hrimg_name)
        filename = filename.split(".")[0]
        stats_lr, stats_hr = self.lr_hr_stat_tuple[idx]
        hr_image = loader(hrimg_name)
        lr_image = loader(lrimg_name)
        lr_unorm = lr_image.copy()
        if self.lognorm:
            image_sign = np.sign(lr_image)
            lr_image = image_sign * np.log(np.abs(lr_image) + 1.0)
            if not self.test:
                image_sign = np.sign(hr_image)
                hr_image = image_sign * np.log(np.abs(hr_image) + 1.0)
        if stats_lr["std"] <= 0.001:
            stats_lr["std"] = 1
        if stats_hr["std"] <= 0.001:
            stats_hr["std"] = 1

        if not self.test:
            hr_image = Normalize()(hr_image, stats_hr)
        lr_image = Normalize()(lr_image, stats_lr)
        sample = {
            "lr": lr_image,
            "lr_un": lr_unorm,
            "hr": hr_image,
            "stats": stats_hr,
            "file": filename,
        }
        if not self.test:
            transforms = Compose(
                [
                    Rotate(),
                    Transpose(),
                    HorizontalFlip(),
                    VerticalFlip(),
                    Pertube(1.00e-6),
                    Reshape(),
                    ToFloatTensor(),
                ]
            )
            for i, trans in enumerate([transforms]):
                sample = trans(sample)
        if self.test:
            transforms = Compose([Pertube(1.00e-6), Reshape(), ToFloatTensor()])
            sample = transforms(sample)
        return sample


class SrDataset(Dataset):
    """Dataset class for loading large amount of image arrays data. This dataset doe not have hr"""

    def __init__(self, root_dir, lognorm=False, test=False, hr=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            lognorm: True if we ar eusing log normalization
            test: True only for test dataset
            hr: Input is hr image, lr is computed, then True
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = Path(root_dir).expanduser().resolve().absolute()
        self.datalist = list(self.root_dir.rglob("*.npz"))
        self.lognorm = lognorm
        self.test = test
        self.hr = hr
        self.statlist = []
        for fname in self.datalist:
            file_path = Path(fname)
            stat_file = json.load(open(str(file_path.parent / "stats.json")))
            self.statlist.append(stat_file)
        print("Total number of data elements found = ", len(self.datalist))

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        img_name = Path(self.datalist[idx])
        filename = os.path.basename(img_name)
        filename = filename.split(".")[0]
        stats = self.statlist[idx]
        if self.hr:
            hr_image = loader(img_name)
        if not self.test:
            if self.lognorm:
                image_sign = np.sign(hr_image)
                hr_image = image_sign * np.log(np.abs(hr_image) + 1.0)
            if stats["std"] <= 0.001:
                stats["std"] = 1
            hr_image = Normalize()(hr_image, stats)

        if self.hr:
            lr_image = scipy.ndimage.zoom(scipy.ndimage.zoom(hr_image, 0.25), 4.0)
        else:
            lr_image = loader(img_name)
            hr_image = np.zeros_like(lr_image)
        if not self.test:
            sample = {
                "lr": lr_image,
                "lr_un": lr_image,
                "hr": hr_image,
                "stats": stats,
                "file": filename,
            }
            transforms = Compose(
                [
                    Differential(),
                    Rotate(),
                    Transpose(),
                    HorizontalFlip(),
                    VerticalFlip(),
                    Pertube(1.00e-6),
                    Reshape(),
                    ToFloatTensor(),
                ]
            )
            for i, trans in enumerate([transforms]):
                sample = trans(sample)
        else:
            if self.lognorm:
                image_sign = np.sign(lr_image)
                lr_image = image_sign * np.log(np.abs(lr_image) + 1.0)
            if stats["std"] <= 0.001:
                stats["std"] = 1
            lr_unorm = lr_image.copy()
            lr_image = Normalize()(lr_image, stats)
            sample = {
                "lr": lr_image,
                "lr_un": lr_unorm,
                "hr": hr_image,
                "stats": stats,
                "file": filename,
            }
            transforms = Compose([Pertube(1.00e-6), Reshape(), ToFloatTensor()])
            sample = transforms(sample)
        return sample


class Rotate:
    """Rotate class rotates image array"""

    def __call__(self, sample):
        """

        Parameters
        ----------
        sample: dictionary containing lr, hr and stats

        Returns
        -------
        sample: dictionary containing transformed lr and transformed hr
        """
        for i in range(random.randint(0, 3)):
            sample["hr"] = np.rot90(sample["hr"])
            sample["lr"] = np.rot90(sample["lr"])

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
        sample["hr"] = torch.tensor(sample["hr"], dtype=torch.float32)
        sample["lr"] = torch.tensor(sample["lr"], dtype=torch.float32)

        return sample


class Transpose:
    """Transpose class calculates the transpose of the matrix"""

    def __call__(self, sample):
        """

        Parameters
        ----------
        sample: dictionary containing lr, hr and stats

        Returns
        -------
        sample: dictionary containing transformed lr and transformed hr
        """
        if random.randint(1, 10) > 5:
            sample["hr"] = np.transpose(sample["hr"])
            sample["lr"] = np.transpose(sample["lr"])

        return sample


class VerticalFlip:
    """VerticalFlip class to probailistically return vertical flip of the matrix"""

    def __call__(self, sample):
        """

        Parameters
        ----------
        sample: dictionary containing lr, hr and stats

        Returns
        -------
        sample: dictionary containing transformed lr and transformed hr
        """
        if random.random() > 0.5:
            sample["hr"] = np.flipud(sample["hr"])
            sample["lr"] = np.flipud(sample["lr"])

        return sample


class HorizontalFlip:
    """HorizontalFlip class to probailistically return horizontal flip of the matrix"""

    def __call__(self, sample):
        """

        Parameters
        ----------
        sample: dictionary containing lr, hr and stats

        Returns
        -------
        sample: dictionary containing transformed lr and transformed hr
        """
        if random.random() > 0.5:
            sample["hr"] = np.fliplr(sample["hr"])
            sample["lr"] = np.fliplr(sample["lr"])

        return sample


class Pertube:
    """ Pertube class transforms image array by adding very small values to the array """

    def __init__(self, episilon=1.00e-10):
        """

        Parameters
        ----------
        episilon: a very small float value
        """
        self.episilon = episilon

    def __call__(self, sample):
        """
        Parameters
        ----------
        sample: dictionary containing lr, hr and stats

        Returns
        -------
        sample: dictionary containing transformed lr and transformed hr
        """

        data = sample["stats"]
        sample["hr"] = sample["hr"] + 0.0
        # sample["lr"] = sample["lr"] + (data["std"] / 100.0) * np.random.rand(*(sample["lr"].shape))
        sample["lr"] = sample["lr"] + 0.0
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

        # setting hr width and height
        hr_width, hr_height = sample["hr"].shape

        # setting lr width and height
        lr_width, lr_height = sample["lr"].shape

        # seeting lr uniform width and height
        lr_un_width, lr_un_height = sample["lr_un"].shape

        sample["hr"] = np.reshape(sample["hr"], (1, hr_width, hr_height))
        sample["lr"] = np.reshape(sample["lr"], (1, lr_width, lr_height))
        sample["lr_un"] = np.reshape(sample["lr_un"], (1, lr_un_width, lr_un_height))
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

class Differential:
    """This will calculate the difference gradient matrix of the image"""
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, sample):
        """

        Parameters
        ----------
        sample
        prob
        Returns
        -------

        """

        if random.random() < self.prob:
            sample["hr"] = np.pad(sample["hr"], 1)[1:, :] - np.pad(sample["hr"], 1)[:-1, :]
            sample["lr"] = np.pad(sample["lr"], 1)[1:, :] - np.pad(sample["lr"], 1)[:-1, :]
        return sample

