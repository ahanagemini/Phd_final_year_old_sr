import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, inChannels, outChannels, kernel =3, stride=1, padding=1):
        '''

        Parameters
        ----------
        inChannels:int
        Number of channels in the input image

        outChannels:int
        Filters (output channels produced by conv)

        kernel:int
        Filter size

        stride: int
        Stride of the convolution

        padding: int
        Zero-padding added to both sides of the input
        '''
        super(Conv, self).__init__()
        self.conv2d = nn.Conv2d(in_channels= inChannels, out_channels= outChannels,
                                kernel_size=kernel, stride=stride, padding=padding)
        self.batchNorm = nn.BatchNorm2D(num_features = outChannels)

    def forward(self, X):
        '''

        Parameters
        ----------
        X: Tensor
        Input


        Returns
        -------
        X: Tensor
        Output
        '''
        X = self.batchNorm(self.conv2d(X))
        X = F.elu(X)
        return X

class ConvTranspose(nn.Module):
    def __init__(self, inChannels, outChannels, kernel, stride, padding, outPadding):
        '''

        Parameters
        ----------
        inChannels: int
        Number of channels in the input image

        outChannels: int
        filters

        kernel: int
        filter size

        stride: int
        Stride of the convolution

        padding: int
         zero-padding will be added to both sides of each dimension in the input

        outPadding: int
         Additional size added to one side of each dimension in the output shape
        '''
        super(ConvTranspose, self).__init__()
        self.convtranspose2D = nn.ConvTranspose2D(in_channels=inChannels, out_channels=outChannels,
                                                  kernel_initializer=kernel, stride=stride, padding=padding,
                                                  output_padding = outPadding)

    def forward(self, X):
        '''

        Parameters
        ----------
        X: Tensor
        Input

        Returns
        -------
        X: Tensor
        Output
        '''
        return self.convtranspose2D(X)

class Downsampler(nn.Module):
    def __init__(self, inChannels, outChannels, poolType='AVG', pool_size=2, poolStride=2):
        '''

        Parameters
        ----------
        inchannels: int
        Number of channels in the input image

        outChannels: int
        filters

        poolType: String
        The type of pooling whether max or avg, Specify in caps MAX or AVG

        pool_size: int
        the size of the window to take pool over
        '''
        super(Downsampler, self).__init__()
        self.num_conv_blocks = 4
        if poolType == "MAX":
            self.pool = nn.MaxPool2D(kernel_size = pool_size, stride = poolStride)
        else:
            self.pool = nn.AvgPool2D(kernel_size = pool_size, stride = poolStride)
        self.conv = Conv(inChannels=inChannels, outChannels=outChannels)

    def forward(self, X):
        '''

        Parameters
        ----------
        X

        Returns
        -------

        '''
        for i in range(len(self.num_conv_blocks)):
            X = self.conv(X)
        return self.pool(X)

class Upsampling(nn.Module):
    def __init__(self, inChannels, outChannels):
        '''

        Parameters
        ----------
        inChannels: int
        Number of channels in the input image

        outChannels: int
        filters
        '''
        self.num_conv_blocks = 2
        self.pixelShuffle = nn.PixelShuffle()
        self.convtranspose = ConvTranspose(inChannels=inChannels, outChannels=inChannels//2)
        self.conv = Conv(inChannels=inChannels, outChannels=outChannels)

    def forward(self, X1, X2):
        '''

        Parameters
        ----------
        X1: Tensor


        X2: Tensor


        Returns
        -------

        '''
        X1
        X1 = self.convtranspose(X1)
        X = torch.cat(X1, X2)
        for i in range(len(self.num_conv_blocks)):
            X = self.conv(X)
        return X

class OutConv(nn.Module):
    def __init__(self, inChannels, outChannels):
        '''

        Parameters
        ----------
        inChannels: int
        Number of channels in the input image

        outChannels: int
         filters
        '''

        self.outconv = Conv(inChannels=inChannels, outChannels=outChannels)
    def forward(self, X):
        '''

        Parameters
        ----------
        X: Tensor

        Returns
        -------

        '''
        return self.outconv(X)

