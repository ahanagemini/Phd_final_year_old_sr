"""
Dataset file
"""
from pathlib import Path
import json
import random
import os

# from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import scipy.ndimage
import numpy as np
from cutter import loader
from skimage.transform import resize

class PairedDataset(Dataset):
    """Dataset class for loading large amount of image arrays data"""

    def __init__(self, root_dir, lognorm=False, test=False):
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
        self.lrlist = sorted(list(self.lr_dir.rglob("*.npz")))
        self.hrlist = sorted(list(self.hr_dir.rglob("*.npz")))
        self.lognorm = lognorm
        self.test = test
        self.hrstatlist = []
        for fname in self.hrlist:
            file_path = Path(fname)
            stat_file = json.load(open(str(file_path.parent / "stats.json")))
            self.hrstatlist.append(stat_file)
        self.lrstatlist = []
        for fname in self.lrlist:
            file_path = Path(fname)
            stat_file = json.load(open(str(file_path.parent / "stats.json")))
            self.lrstatlist.append(stat_file)

        print("Total number of data elements found = ", len(self.hrlist))

    def __len__(self):
        return len(self.hrlist)

    def __getitem__(self, idx):
        hrimg_name = Path(self.hrlist[idx])
        lrimg_name = Path(self.lrlist[idx])
        filename = os.path.basename(hrimg_name)
        filename = filename.split('.')[0]
        stats_hr = self.hrstatlist[idx]
        stats_lr = self.lrstatlist[idx]
        hr_image =  loader(hrimg_name)

        lr_image =  loader(lrimg_name)
        lr_unorm = lr_image.copy()
        if self.lognorm:
            stats_lr = {}
            image_sign = np.sign(lr_image)
            lr_image = image_sign * np.log(np.abs(lr_image) + 1.0)
            stats_lr["mean"] = np.mean(lr_image)
            stats_lr["std"] = np.std(lr_image)
            if not self.test:
                stats_hr = {}
                image_sign = np.sign(hr_image)
                hr_image = image_sign * np.log(np.abs(hr_image) + 1.0)
                stats_hr["mean"] = np.mean(hr_image)
                stats_hr["std"] = np.std(hr_image)
        if stats_lr["std"] <= 0.001:
            stats_lr["std"] = 1
        if stats_hr["std"] <= 0.001:
            stats_hr["std"] = 1

        if not self.test:
            hr_image = Normalize()(hr_image, stats_hr)
        lr_image = Normalize()(lr_image, stats_lr)
        sample = {"lr": lr_image, "lr_un": lr_unorm, "hr": hr_image, "stats": stats_hr, "file": filename}
        if not self.test:
            transforms = Compose(
                [Rotate(), Transpose(), HorizontalFlip(), VerticalFlip(),
                    Pertube(1.00e-6), Reshape(), ToFloatTensor()]
            )
            for i, trans in enumerate([transforms]):
                sample = trans(sample)
        if self.test:
            transforms = Compose(
                [Pertube(1.00e-6), Reshape(), ToFloatTensor()]
            )
            sample = transforms(sample)
        return sample

class AEDataset(Dataset):
    """Dataset class for loading large amount of image arrays data"""

    def __init__(self, root_dir, lognorm=False, test=False):
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
        filename = filename.split('.')[0]
        stats = self.statlist[idx]
        image = loader(img_name)
        #image = image.astype(np.int16)
        # image = scipy.ndimage.zoom(image, 0.25)
        image = resize(image, (64, 64), order=3, preserve_range=True)
        image_unorm = image.copy()
        if self.lognorm:
            image_sign = np.sign(image)
            image = image_sign * np.log(np.abs(image) + 1.0)
            stats = {}
            stats["mean"] = np.mean(image)
            stats["std"] = np.std(image)

        if stats["std"] <= 0.001:
            stats["std"] = 1
        # AE will be used for dimension reduction, so we want reconstruction,
        # LR or dimension reduced HR is both input and target
        # For training and validation normalization input and target
        # For testing, normalize only the input
        image = Normalize()(image, stats)
        if self.test:
            sample = {"lr": image, "hr": image_unorm, "lr_un": image_unorm,
                      "stats": stats, "file": filename}
        else:
            sample = {"lr": image, "hr": image, "lr_un": image_unorm,
                      "stats": stats, "file": filename}
        transforms = Compose([Reshape(), ToFloatTensor()])
        sample = transforms(sample)
        return sample


class SrDataset(Dataset):
    """Dataset class for loading large amount of image arrays data"""

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
        filename = filename.split('.')[0]
        stats = self.statlist[idx]
        if self.hr:
            hr_image = loader(img_name)
        if not self.test:
            if self.lognorm:
                image_sign = np.sign(hr_image)
                hr_image = image_sign * np.log(np.abs(hr_image) + 1.0)
                stats = {}
                stats["mean"] = np.mean(hr_image)
                stats["std"] = np.std(hr_image)

            if stats["std"] <= 0.001:
                stats["std"] = 1
            hr_image = Normalize()(hr_image, stats)
        
        if self.hr:
            lr_image = scipy.ndimage.zoom(scipy.ndimage.zoom(hr_image, 0.25), 4.0)
        else:
            lr_image =  loader(img_name)
            hr_image = np.zeros_like(lr_image)
        if not self.test:
            sample = {"lr": lr_image, "lr_un": lr_image, "hr": hr_image, "stats": stats, "file": filename}
            transforms = Compose(
                [Rotate(), Transpose(), HorizontalFlip(), VerticalFlip(),
                    Pertube(1.00e-6), Reshape(), ToFloatTensor()]
            )
            for i, trans in enumerate([transforms]):
                sample = trans(sample)
        else:
            lr_unorm = lr_image.copy()
            if self.lognorm:
                stats = {}
                image_sign = np.sign(lr_image)
                lr_image = image_sign * np.log(np.abs(lr_image) + 1.0)
                stats["mean"] = np.mean(lr_image)
                stats["std"] = np.std(lr_image)
            if stats["std"] <= 0.001:
                stats["std"] = 1
            lr_image = Normalize()(lr_image, stats)
            sample = {"lr": lr_image, "lr_un": lr_unorm, "hr": hr_image, "stats": stats, "file": filename}
            transforms = Compose(
                [Pertube(1.00e-6), Reshape(), ToFloatTensor()]
            )
            sample = transforms(sample)
        return sample


"""
if __name__ == "__main__":
    face_dataset = SrDataset(root_dir='../data')

    fig = plt.figure()

    for i in range(len(face_dataset)):
        sample = face_dataset[i]

        print(i, sample['hr'].shape, sample['stats'])

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        plt.imshow(sample['lr'])

        if i == 3:
            plt.show()
            break
"""


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
        if random.randint(1, 10) > 5:
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
        if random.randint(1, 10) > 5:
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
        #sample["lr"] = sample["lr"] + (data["std"] / 100.0) * np.random.rand(*(sample["lr"].shape))
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
        width = sample["hr"].shape[-1]
        sample["hr"] = np.reshape(sample["hr"], (1, -1, width))
        sample["lr"] = np.reshape(sample["lr"], (1, -1, width))
        sample["lr_un"] = np.reshape(sample["lr_un"], (1, -1, width))
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
