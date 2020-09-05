import torch.nn.functional as F
import torch.nn as nn
import torch


class ResidualBlock(nn.Module):
    """
    This class implements a residual block, a single ResNet block
    """
    expansion = 1
    def __init__(self, inchannel, outchannel, stride=1):
        """

        Parameters
        ----------
        inchannel:int
        Number of channels in the input image

        outchannel:int
        Filters (output channels produced by conv)

        stride: int
        Stride of the convolution

        """

        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(inchannel, outchannel, kernel_size=3,
                                  stride=stride, padding=1, bias=True),
                        nn.BatchNorm2d(outchannel),
                )
        self.conv2 = nn.Sequential(
                        nn.Conv2d(outchannel, outchannel, kernel_size=3,
                                  stride=1, padding=1, bias=True),
                        nn.BatchNorm2d(outchannel)
                )
        self.skip = nn.Sequential()
        if stride != 1 or inchannel != self.expansion * outchannel:
            self.skip = nn.Sequential(
                nn.Conv2d(inchannel, self.expansion * outchannel, kernel_size=1,
                          stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion * outchannel)
            )

    def forward(self, X):
        """

        Parameters
        ----------
        x: Tensor
        Input


        Returns
        -------
        out: Tensor
        Output
        """

        out = F.relu(self.conv1(X))
        out = self.conv2(out)
        out += self.skip(X)
        out = F.relu(out)
        return out


class AxialNet(nn.Module):
    """
    This class implements the AxialNet or bicubicNet model
    """
    def __init__(self, num_channels, resblocks=2, skip=True):
        """

        Parameters
        ----------
        num_channels: int
        Number of channels in the input and output image

        resblocks: int
        Number of resnet blocks in the model

        skip: boolean
        If there is a skip connection across the axial part of the model
        """

        super(AxialNet, self).__init__()
        self.inchannel = 64
        self.resblocks = resblocks
        self.skip = 1 if skip else 0
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
        )
        self.layer1 = ResidualBlock(64, 64, stride=1)
        if resblocks == 4:
            self.layer1_1 = ResidualBlock(64, 128, stride=1)
        self.layer2 = nn.Conv2d(resblocks * 32, resblocks * 32, kernel_size=(1, 9),
                                padding=(0, 4), stride=(1, 1))
        self.layer3 = nn.Conv2d(resblocks * 32, resblocks * 32, kernel_size=(9, 1),
                                padding=(4, 0), stride=(1, 1))
        if resblocks == 4:
            self.layer4_1 = ResidualBlock(resblocks * 32, 64, stride=1)
        self.layer4 = ResidualBlock(64 + skip * 32 * resblocks, num_channels, stride=1)

        #self.fc = nn.Linear(512*Residu1alBlock.expansion, num_classes)

    def forward(self, x):
        """

        Parameters
        ----------
        x: Tensor
        Input


        Returns
        -------
        out: Tensor
        Output
        """

        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        if self.resblocks == 4:
            out = self.layer1_1(out)
        out1 = self.layer2(out)
        out1 = self.layer3(out1)
        if self.resblocks == 4:
            out1 = self.layer4_1(out1)
        if self.skip == 1:
            out2 = torch.cat((out, out1), 1)
        else:
            out2 = out1
        out = self.layer4(out2)
        return out

