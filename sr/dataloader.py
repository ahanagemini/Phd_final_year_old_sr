import torch

class DataSet(torch.utils.data.Dataset):
    def __init__(self, xTrain, yTrain):
        self.yTrain = yTrain
        self.xTrain = xTrain

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self):
        X = torch.load(self.xTrain+ '.npy')
        y = torch.load(self.yTrain+ '.npy')
        return X,y


