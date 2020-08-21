from pathlib import Path
import json
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset

from cutter import Loader

class SrDataset(Dataset):
    def __init__(self,root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = Path(root_dir)
        self.datalist = list(self.root_dir.rglob('*.npz'))
        self.statlist = []
        for fname in self.datalist:
            p = Path(fname)
            d = json.load(open(str(p.parent / "stats.json")))
            self.statlist.append(d)
        print("Total number of data elements found = ", len(self.datalist))
        self.transform = transform

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = Path(self.datalist[idx])
        stats = self.statlist[idx]
        hr = Loader(img_name)

        sample = {'hr': hr, 'lr': lr, 'stats': stats}

        if self.transform:
            sample = self.transform(sample)

        return sample

if __name__ == "__main__":
    face_dataset = SrDataset(root_dir='../data')

    fig = plt.figure()

    for i in range(len(face_dataset)):
        sample = face_dataset[i]

        print(i, sample['image'].shape, sample['stats'])

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        plt.imshow(sample['image'])

        if i == 3:
            plt.show()
            break