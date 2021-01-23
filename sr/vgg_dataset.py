from pathlib import Path
import json
import random
import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import numpy as np
from stat_plotter import PlotStat
from cutter import loader


class VGGTrainDataset(Dataset):
    """Dataset class for loading large amount of image arrays to train vgg"""

    def __init__(self, root_dir, lognorm=False, validate=True, test=False):
        self.root_dir = Path(root_dir).expanduser().resolve().absolute()
        self.lr_dir = self.root_dir / "LR"
        self.hr_dir = self.root_dir / "HR"
        self.image_label_stat_tuple_list = []
        self.lr_hr_tuple = []
        self.stats_list = []
        for image_file_name in self.lr_dir.rglob("*.npz"):
            image_name = image_file_name.name
            image_parent_name = os.path.splitext(image_file_name.parent.name)[0]
            lr_image = image_file_name
            lr_image_label = 0.0
            hr_image = self.hr_dir / image_parent_name / image_name
            hr_image_label = 1.0
            self.lr_hr_tuple.append((lr_image, lr_image_label))
            self.lr_hr_tuple.append((hr_image, hr_image_label))

        np.random.shuffle(self.lr_hr_tuple)
        self.lognorm = lognorm
        self.test = test
        self.validate = validate
        keys = ("mean", "std", "min", "max")
        for ftuple in self.lr_hr_tuple:
            image, label = ftuple
            hstat = json.load(open(str(image.parent / "stats.json")))
            hstat = {x: hstat[x] for x in keys}
            self.stats_list.append(hstat)

    def __len__(self):
        return len(self.lr_hr_tuple)

    def __getitem__(self, idx):
        image_path, image_label = self.lr_hr_tuple[idx]
        image_name = Path(image_path)
        filename = os.path.basename(image_name)
        filename = filename.split(".")[0]
        stats_hr = self.stats_list[idx]
        image = loader(image_path)
        image = Normalize()(image, stats_hr)
        sample = {"image": image, "label": image_label, "stats": stats_hr, "file": filename}
        transforms = Compose([Differential(0.4),
                              Rotate(),
                              Transpose(),
                              HorizontalFlip(),
                              VerticalFlip(),
                              Pertube(0.5, 0.5),
                              Reshape(),
                              ToFloatTensor()])
        for i, trans in enumerate([transforms]):
            sample = trans(sample)

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
        image_height_lr, image_width_lr = sample["image"].shape
        sample["image"] = np.reshape(sample["image"], (1, image_height_lr, image_width_lr))
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
        sample["image"] = torch.from_numpy(sample["image"].copy()).float()
        return sample

class Differential:
    """This will calculate the difference gradient matrix of the image"""
    def __init__(self, prob):
        """

        Parameters
        ----------
        prob: The probability of this filter running between 0 and 1
        """
        self.prob = prob

    def __call__(self, sample):
        """

        Parameters
        ----------
        sample: Contains lr, hr images
        Returns
        -------

        """
        if random.random() < self.prob:
            lr_height, lr_width = sample["image"].shape
            sample["image"] = np.abs(np.diff(np.pad(sample["image"], 1))[1: 1+lr_width, 0:lr_height])
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
            sample["image"] = np.rot90(sample["image"])
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
        if random.random() > 0.5:
            sample["image"] = np.transpose(sample["image"])
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
            sample["image"] = np.flipud(sample["image"])
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
            sample["image"] = np.fliplr(sample["image"])
        return sample


class Pertube:
    """ Pertube class transforms image array by adding very small values to the array """

    def __init__(self, prob=0.5, random_prob=0.5):
        """

        Parameters
        ----------
        prob: probabilty of no. of images to be processed using this filter
        random_prob: the probability of generating zeros in an array
        """
        self.prob = prob
        self.random_prob = random_prob

    def __call__(self, sample):
        """
        Parameters
        ----------
        sample: dictionary containing lr, hr and stats

        Returns
        -------
        sample: dictionary containing transformed lr and transformed hr
        """

        if random.random() < self.prob:
            lr_width, lr_height = sample["image"].shape
            sample["image"] = np.random.choice([0, 1], size=(lr_width, lr_height),
                                                      p=[self.random_prob, 1-self.random_prob]) * sample["image"]
        return sample