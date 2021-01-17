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
        keys = ("mean", "std", "min", "max")
        # After that randomly shuffle those pairs
        for image_file_name in self.lr_dir.rglob("*.npz"):
            parent_file = image_file_name.parent
            label = 1
            stat = json.load(open(str(parent_file / "stats.json")))
            stat = {x: stat[x] for x in keys}
            self.image_label_stat_tuple_list.append((image_file_name, label, stat))

        for image_file_name in self.hr_dir.rglob("*.npz"):
            parent_file = image_file_name.parent
            label = 0
            stat = json.load(open(str(parent_file / "stats.json")))
            stat = {x: stat[x] for x in keys}
            self.image_label_stat_tuple_list.append((image_file_name, label, stat))

        random.shuffle(self.image_label_stat_tuple_list)
        self.lognorm = lognorm
        self.test = test
        self.validate = validate

    def __len__(self):
        return len(self.image_label_stat_tuple_list)

    def __getitem__(self, idx):
        image_name, label, stat = self.image_label_stat_tuple_list[idx]
        filename = os.path.splitext(image_name.name)[0]
        image = loader(image_name)
        image = Normalize()(image, stat)

        sample = {"image": image, "stats": stat, "filename": filename, "label": label}

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
        image_height, image_width = sample["image"].shape

        sample["image"] = np.reshape(sample["image"], (1, image_height, image_width))
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
        sample["image"] = torch.from_numpy(sample["image"]).float()
        return sample