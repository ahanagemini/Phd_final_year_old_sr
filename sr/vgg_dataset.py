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
            hr_image = self.hr_dir / image_parent_name / image_name
            self.lr_hr_tuple.append((lr_image, hr_image))

        np.random.shuffle(self.lr_hr_tuple)
        self.lognorm = lognorm
        self.test = test
        self.validate = validate
        keys = ("mean", "std", "min", "max")
        for ftuple in self.lr_hr_tuple:
            lr_image, hr_image = ftuple
            hstat = json.load(open(str(hr_image.parent / "stats.json")))
            hstat = {x: hstat[x] for x in keys}
            self.stats_list.append(hstat)

    def __len__(self):
        return len(self.lr_hr_tuple)

    def __getitem__(self, idx):
        lrimg_name, hrimg_name = self.lr_hr_tuple[idx]
        lrimg_name = Path(lrimg_name)
        hrimg_name = Path(hrimg_name)
        filename = os.path.basename(hrimg_name)
        filename = filename.split(".")[0]
        stats_hr = self.stats_list[idx]
        hr_image = loader(hrimg_name)
        lr_image = loader(lrimg_name)

        lr_image = Normalize()(lr_image, stats_hr)
        hr_image = Normalize()(hr_image, stats_hr)
        sample = {"lr": lr_image, "hr": hr_image, "stats": stats_hr, "filename": filename}

        transforms = Compose([Reshape(), ToFloatTensor()])
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
        image_height_lr, image_width_lr = sample["lr"].shape
        image_height_hr, image_width_hr = sample["hr"].shape

        sample["lr"] = np.reshape(sample["lr"], (1, image_height_lr, image_width_lr))
        sample["hr"] = np.reshape(sample["hr"], (1, image_height_hr, image_width_hr))
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
        sample["lr"] = torch.from_numpy(sample["lr"]).float()
        sample["hr"] = torch.from_numpy(sample["hr"]).float()
        return sample
