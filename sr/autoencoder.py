import os
import torch 
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image

class AE(nn.Module):
    def __init__(self, num_channels):
        super(AE, self).__init__()
        # encoder layers
        self.enc1 = nn.Conv2d(num_channels, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.enc4 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        # self.enc5 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # decoder layers
        # self.dec0 = nn.ConvTranspose2d(8, 8, kernel_size=2, stride=1, padding=1)
        self.dec1 = nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2)
        self.dec2 = nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2)
        self.dec3 = nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2)
        self.dec4 = nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2)
        self.out = nn.Conv2d(64, num_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # encode
        x = F.relu(self.enc1(x))
        x = self.pool(x)
        x = F.relu(self.enc2(x))
        x = self.pool(x)
        x = F.relu(self.enc3(x))
        x = self.pool(x)
        x = F.relu(self.enc4(x))
        enc = self.pool(x) 
        #enc = F.relu(self.enc5(x))
        #enc = self.pool(x)
        # the latent space representation

        # decode
        # res = F.relu(self.dec0(enc))
        res = F.relu(self.dec1(enc))
        res = F.relu(self.dec2(res))
        res = F.relu(self.dec3(res))
        res = F.relu(self.dec4(res))
        res = self.out(res)
        return enc, res

class AE2(nn.Module):
    def __init__(self, num_channels=3):
        super(AE, self).__init__()
        # encoder layers
        self.enc1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=2, padding=1)
        self.enc2 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1)
        self.enc4 = nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1)
        self.enc5 = nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1)

        # decoder layers
        self.dec0 = nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2)
        self.dec1 = nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2)
        self.dec2 = nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2)
        self.dec3 = nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2)
        self.dec4 = nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2)
        self.out = nn.Conv2d(64, num_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # encode
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        enc = F.relu(self.enc5(x))
        # decode
        res =  F.relu(self.dec0(enc))
        res = F.relu(self.dec1(res))
        res = F.relu(self.dec2(res))
        res = F.relu(self.dec3(res))
        res = F.relu(self.dec4(res))
        res = self.out(res)
        return enc, res
