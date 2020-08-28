import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    '''This class performs the Convolution Operation'''
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=1):
        '''

        Parameters
        ----------
        in_channels:int
        Number of channels in the input image

        out_channels:int
        Filters (output channels produced by conv)

        kernel:int
        Filter size

        stride: int
        Stride of the convolution

        padding: int
        Zero-padding added to both sides of the input
        '''
        super(Conv, self).__init__()
        self.conv_2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        '''

        Parameters
        ----------
        x: Tensor
        Input


        Returns
        -------
        x: Tensor
        Output
        '''
        x = self.batch_norm(self.conv_2d(x))
        x = F.elu(x)
        return x

class ConvTranspose(nn.Module):
    '''This class is for Upsampling using conv transpose'''
    def __init__(self, in_channels, out_channels, kernel=2, stride=2):
        '''

        Parameters
        ----------
        in_channels: int
        Number of channels in the input image

        out_channels: int
        filters

        kernel: int
        filter size

        stride: int
        Stride of the convolution
        '''
        super(ConvTranspose, self).__init__()
        self.conv_transpose_2d = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                                    kernel_size=kernel, stride=stride)

    def forward(self, x):
        '''

        Parameters
        ----------
        x: Tensor
        Input

        Returns
        -------
        x: Tensor
        Output
        '''
        return self.conv_transpose_2d(x)

class Downsampler(nn.Module):
    '''This class is for performing downsampling'''
    def __init__(self, in_channels, out_channels, pool_type='AVG', pool_size=2, pool_stride=2):
        '''

        Parameters
        ----------
        in_channels: int
        Number of channels in the input image

        out_channels: int
        filters

        pool_type: String
        The type of pooling whether max or avg, Specify in caps MAX or AVG

        pool_size: int
        the size of the window to take pool over
        '''
        super(Downsampler, self).__init__()
        self.num_conv_blocks = 4
        if pool_type == "MAX":
            self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride)
        else:
            self.pool = nn.AvgPool2d(kernel_size=pool_size, stride=pool_stride)
        self.conv = Conv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x):
        '''

        Parameters
        ----------
        x: tensor
        Input

        Returns
        -------

        '''
        for i in range(len(self.num_conv_blocks)):
            x = self.conv(x)
        return x, self.pool(x)


class UnetBase(nn.Module):
    '''this class is for Unet bottom base operations'''
    def __init__(self, in_channels, out_channels, pool_type="AVG", pool_size=2, pool_stride=2):
        '''

        Parameters
        ----------
        in_channels: int
        Number of channels in the input image

        out_channels: int
        filters

        pool_type: String
        The type of pooling whether max or avg, Specify in caps MAX or AVG

        pool_size:
        the size of the window to take pool over
        '''
        super(UnetBase, self).__init__()
        self.num_conv_blocks = 3
        if pool_type == "MAX":
            self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride)
        else:
            self.pool = nn.AvgPool2d(kernel_size=pool_size, stride=pool_stride)
        self.conv = Conv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x):
        for i in range(self.num_conv_blocks):
            x = self.conv(x)
        return x


class Upsampling(nn.Module):
    '''this class is for upsampling'''
    def __init__(self, in_channels, out_channels):
        '''

        Parameters
        ----------
        in_channels: int
        Number of channels in the input image

        out_channels: int
        filters
        '''
        super(Upsampling, self).__init__()
        self.num_conv_blocks = 2
        self.conv_transpose = ConvTranspose(in_channels=in_channels, out_channels=in_channels//2)
        self.conv = Conv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x_1, x_2):
        '''

        Parameters
        ----------
        x_1: Tensor
        Input

        x_2: Tensor
        Skip

        Returns
        -------

        '''
        x_1 = self.conv_transpose(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        for i in range(len(self.num_conv_blocks)):
            x = self.conv(x)
        return x

class OutConv(nn.Module):
    '''This class will give the output'''
    def __init__(self, in_channels, out_channels):
        '''

        Parameters
        ----------
        in_channels: int
        Number of channels in the input image

        out_channels: int
         filters
        '''
        super(OutConv, self).__init__()
        self.out_conv = Conv(in_channels=in_channels, out_channels=out_channels)
        
    def forward(self, x):
        '''

        Parameters
        ----------
        x: Tensor
        Input

        Returns
        -------
        x: Tensor
        output
        '''
        return self.out_conv(x)

class UNET(nn.Module):
    '''Unet Structure'''
    def __init__(self, inchannels=3, outchannels=3, init_features=32):
        super(UNET, self).__init__()

        #Downsampler
        self.enc1 = Downsampler(in_channels=inchannels, out_channels=init_features)
        self.enc2 = Downsampler(in_channels=init_features, out_channels=init_features*2)
        self.enc3 = Downsampler(in_channels=init_features*2, out_channels=init_features*4)

        #Base
        self.ubase = UnetBase(in_channels=init_features*4, out_channels=init_features*8)

        #Upsampler
        self.dec3 = Upsampling(in_channels=init_features*8, out_channels=init_features*4)
        self.dec2 = Upsampling(in_channels=init_features*4, out_channels=init_features*2)
        self.dec1 = Upsampling(in_channels=init_features*2, out_channels=init_features)

        #Output
        self.output = OutConv(in_channels=init_features, out_channels=outchannels)

    def forward(self, x):
        '''

        Parameters
        ----------
        x: Tensor
        Input

        Returns
        -------
        x: Tensor
        Output
        '''
        # downsample
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)

        # base
        x = self.ubase(x)

        # upsample
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        # output
        x = self.output(x)
        return x
