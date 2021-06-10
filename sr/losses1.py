import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchsummary import summary
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


class SSIM(nn.Module):
    """
    SSIM Loss
    Modified from https://github.com/huster-wgm/Pytorch-metrics/blob/master/metrics.py
    """

    def __init__(self):
        super(SSIM, self).__init__()

    def forward(self, y_pred, y_true):
        ssim_loss = 1 - ssim( y_pred, y_true, data_range=256, size_average=True)
        return ssim_loss


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19
    """

    def __init__(self):
        super().__init__()
        # self.output_layer = output_layer
        self.pretrained = models.vgg19(pretrained=True)
        self.layers = list(self.pretrained._modules['features'][:29])
        self.net = nn.Sequential(*self.layers)
        self.pretrained = None
        for param in net.parameters():
            param.require_grad = False

    def normalize_tensor_transform():
        return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    def forward(self, x, y):

        # Normalizing for input to VGG19
        x = x / 256.0
        y = y / 256.0
        x0 = (x - 0.485) / 0.229
        x1 = (x - 0.456) / 0.224
        x2 = (x - 0.406) / 0.225
        y0 = (y - 0.485) / 0.229
        y1 = (y - 0.456) / 0.224
        y2 = (y - 0.406) / 0.225

        # Converting to 3 channel
        X = torch.cat((x0, x1, x2), dim=1)
        Y = torch.cat((y0, y1, y2), dim=1)

        # Executing VGG19 till 5,4 layer as in SRGAN
        X = self.net(X)
        Y = self.net(Y)

        # Computing perceptual loss
        perceptual_loss = torch.mean((X - Y) ** 2)
        percep_loss = torch.nn.functional.mse_loss(X, Y, reduction='mean')
        print(perceptual_loss, percep_loss)
        return perceptual_loss

