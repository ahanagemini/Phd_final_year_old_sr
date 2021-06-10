import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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

