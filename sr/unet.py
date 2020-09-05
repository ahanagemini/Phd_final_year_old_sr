import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    """This class performs the Convolution Operation"""

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
        super(Conv, self).__init__()
        self.section = []
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
        #self.section.append(nn.GroupNorm2d(num_features=out_channels))

        self.conv = nn.Sequential(*self.section)

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
                #nn.Upsample(mode="bilinear", scale_factor=2),
                nn.Upsample(mode="bicubic", scale_factor=2),
                nn.Conv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )
        self.up_conv = nn.Sequential(
            Conv(in_channels=in_channels, out_channels=out_channels),
            Conv(in_channels=out_channels, out_channels=out_channels),
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

    def __init__(self, in_channels, out_channels, init_features=32, depth=4):
        super(UNET, self).__init__()
        self.downsample = nn.ModuleList()
        self.initial_channels = in_channels
        self.final_channel = out_channels
        for i in range(depth):
            self.downsample.append(
                Conv(
                    in_channels=self.initial_channels,
                    out_channels=init_features * (2 ** i),
                )
            )
            self.initial_channels = init_features * (2 ** i)
        self.upsample = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.upsample.append(
                Upsampling(
                    in_channels=self.initial_channels,
                    out_channels=init_features * (2 ** i), sample_type="upsamp"
                )
            )
            self.initial_channels = init_features * (2 ** i)
        self.out_conv = nn.Conv2d(
            in_channels=self.initial_channels, out_channels=out_channels, kernel_size=1
        )

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
        blocks = []
        for i, down in enumerate(self.downsample):
            x = down(x)
            if i != len(self.downsample) - 1:
                blocks.append(x)
                x = F.avg_pool2d(x, 2)
        for i, up in enumerate(self.upsample):
            x = up(x, blocks[-i - 1])
        return self.out_conv(x)
