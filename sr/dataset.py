from pathlib import Path
from torch.utils.data import Dataset

class SrDataset(Dataset):
	def __init__(self,root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = Path(root_dir)
        self.datalist = self.root_dir.rglob('*.npz')
        self.statlist = []
        for fname in self.datalist:
        	p = Path(fname)
        	d = json.load(p.parent // "stats.json")
        	self.statlist.append(d)
        self.transform = transform

	def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample