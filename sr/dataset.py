"""
Dataset file
"""
from pathlib import Path
import json
import random

# from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import scipy.ndimage
import numpy as np
from cutter import loader


class SrDataset(Dataset):
    """Dataset class for loading large amount of image arrays data"""

    def __init__(self, root_dir, test=False, hr=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            test: True if this code is for testing dataset
            hr: Input is hr image, lr is computed, then True
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = Path(root_dir)
        self.datalist = list(self.root_dir.rglob("*.npz"))
        self.test = test
        self.hr = hr
        self.statlist = []
        for fname in self.datalist:
            file_path = Path(fname)
            if not self.test:
                stat_file = json.load(open(str(file_path.parent / "stats.json")))
                self.statlist.append(stat_file)
        print("Total number of data elements found = ", len(self.datalist))

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        img_name = Path(self.datalist[idx])
        if not self.test:
            stats = self.statlist[idx]
        if self.hr:
            hr_image = loader(img_name)
        if not self.test:
            image_sign = np.sign(hr_image)
            hr_image = image_sign * np.log(np.abs(hr_image) + 1.0)
            stats = {}
            stats["mean"] = np.mean(hr_image)
            stats["std"] = np.std(hr_image)
            hr_image = Normalize()(hr_image, stats)
        # upper_quartile = stats['upper_quartile']
        # lower_quartile = stats['lower_quartile']
        # hr_image[hr_image > upper_quartile] = upper_quartile
        # hr_image[hr_image < lower_quartile] = lower_quartile
        # interval_length = upper_quartile - lower_quartile
        # hr_image -= lower_quartile
        # hr_image /= abs(interval_length)
        # hr_image = (hr_image - 0.5)*2.0
        if self.hr:
            lr_image = scipy.ndimage.zoom(scipy.ndimage.zoom(hr_image, 0.5), 2.0)
        else:
            lr_image =  loader(img_name)
            hr_image = np.zeros_like(lr_image)
        if not self.test:
            sample = {"lr": lr_image, "hr": hr_image, "stats": stats}
            transforms = Compose(
                [Rotate(), Transpose(), Pertube(1.00e-6), Reshape(), ToFloatTensor()]
            )
            for i, trans in enumerate([transforms]):
                sample = trans(sample)
        else:
            stat = {}
            image_sign = np.sign(lr_image)
            lr_image = image_sign * np.log(np.abs(lr_image) + 1.0)
            stats["mean"] = np.mean(lr_image)
            stats["std"] = np.std(lr_image)
            lr_image = Normalize()(lr_image, stats)
            sample = {"lr": lr_image, "hr": hr_image, "stats": stats}
            transforms = Compose(
                [ToFloatTensor()]
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
        sample["hr"] = np.transpose(sample["hr"])
        sample["lr"] = np.transpose(sample["lr"])

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
        sample["lr"] = sample["lr"] + (data["std"] / 100.0) * np.random.rand(*(sample["lr"].shape))
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

        sample["hr"] = np.reshape(sample["hr"], (1, 256, 256))
        sample["lr"] = np.reshape(sample["lr"], (1, 256, 256))
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
