import torch
import torch.nn as nn
import torchvision


class Discriminator_VGG_128(nn.Module):
    def __init__(self, in_nc, nf=16, shape=256):
        super(Discriminator_VGG_128, self).__init__()
        # [1, 256, 256]
        self.conv0_0 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(nf, nf, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(nf, affine=True)
        # [16, 128, 128]
        self.conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(nf * 2, affine=True)
        self.conv1_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(nf * 2, affine=True)
        # [32, 64, 64]
        self.conv2_0 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(nf * 4, affine=True)
        self.conv2_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(nf * 4, affine=True)

        '''
        # [256, 32, 32]
        self.conv3_0 = nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv3_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(nf * 8, affine=True)
        # [512, 16, 16]
        self.conv4_0 = nn.Conv2d(nf * 8, nf * 16, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(nf * 16, affine=True)
        self.conv4_1 = nn.Conv2d(nf * 16, nf * 16, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(nf * 16, affine=True)
        '''
        self.linear1 = nn.Linear(64 * shape//8 * shape//8, 4096)
        self.linear2 = nn.Linear(4096, 512)
        self.linear3 = nn.Linear(512, 2)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        fea = self.lrelu(self.conv0_0(x))
        fea = self.lrelu(self.bn0_1(self.conv0_1(fea)))

        fea = self.lrelu(self.bn1_0(self.conv1_0(fea)))
        fea = self.lrelu(self.bn1_1(self.conv1_1(fea)))

        fea = self.lrelu(self.bn2_0(self.conv2_0(fea)))
        fea = self.lrelu(self.bn2_1(self.conv2_1(fea)))

        fea = fea.view(fea.size(0), -1)
        fea = self.lrelu(self.linear1(fea))
        fea = self.linear2(fea)
        fea = self.log_softmax(self.linear3(fea))
        return fea