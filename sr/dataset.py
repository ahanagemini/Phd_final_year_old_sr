"""
Dataset file
"""
from pathlib import Path
import json

# from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import scipy.ndimage
import numpy as np
from cutter import loader


class SrDataset(Dataset):
    """Dataset class for loading large amount of image arrays data"""

    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = Path(root_dir)
        self.datalist = list(self.root_dir.rglob("*.npz"))
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
        stats = self.statlist[idx]
        hr_image = loader(img_name)
        image_sign = np.sign(hr_image)
        hr_image = image_sign * np.log(np.abs(hr_image) + 1.0)
        # upper_quartile = stats['upper_quartile']
        # lower_quartile = stats['lower_quartile']
        # hr_image[hr_image > upper_quartile] = upper_quartile
        # hr_image[hr_image < lower_quartile] = lower_quartile
        # interval_length = upper_quartile - lower_quartile
        # hr_image -= lower_quartile
        # hr_image /= abs(interval_length)
        # hr_image = (hr_image - 0.5)*2.0
        lr_image = scipy.ndimage.zoom(scipy.ndimage.zoom(hr_image, 0.5), 2.0)
        hr_image = np.reshape(hr_image, (1, 256, 256))
        lr_image = np.reshape(lr_image, (1, 256, 256))
        sample = {"lr": lr_image, "hr": hr_image, "stats": stats}
        transforms = Compose([Rotate(), Transpose(), Pertube(1.00e-6), Reshape(), ToFloatTensor()])
        for i, trans in enumerate([transforms]):
            sample = trans(sample)
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


class Rotate(object):
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
        sample["hr"] = np.rot90(sample["hr"])
        sample["lr"] = np.rot90(sample["lr"])

        return sample


class ToFloatTensor(object):
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


class Transpose(object):
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


class Pertube(object):
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
        sample["hr"] = sample["hr"] + (data["std"] / 100 + self.episilon)
        sample["lr"] = sample["lr"] + (data["std"] / 100 + self.episilon)
        return sample

class Reshape(object):
    '''Reshaping tensors'''

    def __call__(self, sample):
        '''

        Parameters
        ----------
        sample: dictionary containing lr, hr and stats

        Returns
        -------
        sample: dictionary containing reshaped lr and reshaped hr
        '''

        sample["hr"] = np.reshape(sample["hr"], (1, 256, 256))
        sample["lr"] = np.reshape(sample["lr"], (1, 256, 256))
        return sample