import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from .attention import PyramidAttention
import numpy as np

class Final_Upsample(nn.Module):
    """This class performs the final upsampling to produce a higher resolution image"""

    def __init__(self,in_channels, out_channels, kernel=3, stride=1,
                 padding=1, upsample_debt=2, mode="bicubic", scale_factor=2, align_corners=True):
        """

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

        upsample_debt: int
        Length of the number of upsamples that need to be done

        mode: String
        The type of interpolation required during upsampling

        scale_factor: int
        The upsampling scale

        align_corners: bool
        """
        super(Final_Upsample, self).__init__()
        self.section = nn.ModuleList()
        self.section.append(Resnet(in_channels=in_channels, out_channels=4*out_channels,
                                   kernel=kernel, stride=stride, padding=padding))
        self.section.append(nn.PixelShuffle(2))
        self.upsample_conv = nn.Sequential(*self.section)
        self.upsample_debt = upsample_debt

    def forward(self, x):
        """
        Parameters
        ----------
        x
        Returns
        -------
        """
        for i in range(self.upsample_debt):
            x = self.upsample_conv(x)
        return x

class Resnet(nn.Module):
    """This is resnet class"""

    def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=1):
        """

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
        """
        super(Resnet, self).__init__()
        self.conv_section = nn.Sequential(
            Conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=kernel,
                stride=stride,
                conv_section_type="resnet",
            )
        )
        self.identity = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
            )
        )
        self.leaky_relu = nn.Sequential(nn.LeakyReLU())

    def forward(self, x):
        """

        Parameters
        ----------
        x: Tensor
        returns the x value

        Returns
        -------
        out: Tensor
        returns the out value after adding the input
        """
        out = self.conv_section(x)
        identity_x = self.identity(x)
        out = self.leaky_relu(out + identity_x)
        return out


class Conv(nn.Module):
    """This class performs the Convolution Operation"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=3,
        stride=1,
        padding=1,
        conv_section_type="conv",
    ):
        """

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

        conv_section_type: string
        Two options conv and resnet
        """
        super(Conv, self).__init__()
        self.section = nn.ModuleList()
        if conv_section_type == "conv":

            self.section.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                )
            )

            self.section.append(nn.BatchNorm2d(num_features=out_channels))
            self.section.append(
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                )
            )
            self.section.append(nn.BatchNorm2d(num_features=out_channels))
            self.section.append(
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                )
            )
            self.section.append(nn.LeakyReLU())
            self.section.append(nn.BatchNorm2d(num_features=out_channels))

        elif conv_section_type == "resnet":
            self.section.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                )
            )
            self.section.append(nn.LeakyReLU())
            self.section.append(nn.BatchNorm2d(num_features=out_channels))
            self.section.append(
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                )
            )
            self.section.append(nn.BatchNorm2d(num_features=out_channels))
            # self.section.append(nn.GroupNorm2d(num_features=out_channels))

        self.conv = nn.Sequential(*self.section)

    def forward(self, x, dummy=0.0):
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
        out = self.conv(x)
        return out


class Upsampling(nn.Module):
    """this class is for upsampling"""

    def __init__(self, in_channels, out_channels, sample_type="upconv"):
        """

        Parameters
        ----------
        in_channels: int
        Number of channels in the input image

        out_channels: int
        Number of channels in the output image

        sample_type: string
        The type of sampling required if upconv then convtranspose and upsamp then upsample
        filters
        """
        super(Upsampling, self).__init__()
        if sample_type == "upconv":
            self.up_sample = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=2,
                stride=2,
            )
        elif sample_type == "upsamp":
            self.up_sample = self.up = nn.Sequential(
                Resnet(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel=3,
                    stride=1,
                    padding=1,
                ),
                nn.Conv2d(in_channels=out_channels, out_channels=4 * out_channels,
                          kernel_size=3, stride=1, padding=1),
                nn.PixelShuffle(2),
            )
        self.up_conv = nn.Sequential(
            Resnet(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=3,
                stride=1,
                padding=1,
            ),
            Resnet(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel=3,
                stride=1,
                padding=1,
            ),
        )

    def forward(self, input_tensor, skip):
        """

        Parameters
        ----------
        input_tensor: Tensor
        Input

        skip: Tensor
        Skip

        Returns
        -------

        """

        input_tensor = self.up_sample(input_tensor)
        x = torch.cat((input_tensor, skip), dim=1)
        out = self.up_conv(x)
        return out


class UNET(nn.Module):
    """Unet Structure"""

    def __init__(self, in_channels, out_channels, init_features=32, depth=4, k=1):
        super(UNET, self).__init__()
        self.downsample = nn.ModuleList()
        self.initial_channels = in_channels
        self.resnet_out_channels = 32
        self.final_channel = out_channels

        # resnet
        self.resnet_blocks_left = []
        for i in range(k):
            self.resnet_blocks_left.append(
                Resnet(
                    in_channels=self.initial_channels,
                    out_channels=self.resnet_out_channels,
                )
            )
        self.resnet_left = nn.Sequential(*self.resnet_blocks_left)
        self.initial_channels = self.resnet_out_channels
        for i in range(depth):
            if i == depth - 1:
                print(f"init_features is {init_features*(2**i)}")
                self.downsample.append(
                    nn.Sequential(
                        Resnet(
                            in_channels=self.initial_channels,
                            out_channels=init_features * (2 ** i),
                        ),
                        PyramidAttention(channel=init_features * (2 ** i),
                                         reduction=8, res_scale=0.1
                                         ),

                        Resnet(
                            in_channels=init_features * (2 ** i),
                            out_channels=init_features * (2 ** i),
                        ),
                        Resnet(
                            in_channels=init_features * (2 ** i),
                            out_channels=init_features * (2 ** i),
                        ))
                )
            else:
                self.downsample.append(
                    nn.Sequential(
                        Resnet(
                            in_channels=self.initial_channels,
                            out_channels=init_features * (2 ** i),
                        ),
                    Resnet(
                        in_channels=init_features * (2 ** i),
                        out_channels=init_features * (2 ** i),
                    ),
                    Resnet(
                        in_channels=init_features * (2 ** i),
                        out_channels=init_features * (2 ** i),
                    ))
                )
            self.initial_channels = init_features * (2 ** i)

        self.upsample = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.upsample.append(
                Upsampling(
                    in_channels=self.initial_channels,
                    out_channels=init_features * (2 ** i),
                    sample_type="upsamp",
                )
            )
            self.initial_channels = init_features * (2 ** i)
        self.resnet_blocks_right = []
        for i in range(k):
            self.resnet_blocks_right.append(
                Resnet(
                    in_channels=self.initial_channels,
                    out_channels=self.initial_channels,
                )
            )
        self.resnet_right = nn.Sequential(*self.resnet_blocks_right)
        self.final_upsample = Final_Upsample(in_channels=self.initial_channels, out_channels=self.initial_channels,
                                             kernel=3, stride=1, padding=1, upsample_debt=2, mode="bicubic",
                                             scale_factor=2, align_corners=True)

        self.out_conv = nn.Conv2d(
            in_channels=self.initial_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

    def downsampling(self, x):
        """

        Parameters
        ----------

        x: Tensor

        dummy: satisfy checkpoint requirement
        Input

        Returns:
        ----------

        x: output after downsample
        skips: List of skips
        """
        skips = []
        for i, down in enumerate(self.downsample):
            x = down(x)
            if i != len(self.downsample) - 1:
                skips.append(x)
                x = F.avg_pool2d(x, 2)
        return x, skips

    def upsampling(self, x, skips):
        """

        Parameters
        -----------

        x: Tensor
        Input

        skips:List
        contains skips of each layer

        dummy: satisfy checkpoint requirement
        Returns
        -----------
        x: output after upsample
        """
        for i, up in enumerate(self.upsample):
            x = up(x, skips[-i - 1])
        return x

    def forward(self, x):
        """

        Parameters
        ----------
        x: Tensor
        Input

        down_depth: int
        Downsample depth

        Returns
        -------
        x: Tensor
        Output
        """
        # downsample
        x = self.resnet_left(x)
        x, skips = self.downsampling(x)
        x = self.upsampling(x, skips)
        x = self.final_upsample(x)
        x = self.resnet_right(x)
        return self.out_conv(x)
