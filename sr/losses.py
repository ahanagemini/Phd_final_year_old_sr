import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def binarize(y, threshold):
    y[y < threshold] = 0.0
    y[y >= threshold] = 1.0
    return y


class L1loss(nn.Module):
    """L1 Loss"""

    def __init__(self):
        super(L1loss, self).__init__()

    def forward(self, y_pred, y_true):
        """

        Parameters
        ----------
        y_pred
        y_true

        Returns
        -------
        l1 loss
        """
        return torch.mean(torch.abs(y_pred - y_true))


class SSIM(nn.Module):
    """
    SSIM Loss
    Modified from https://github.com/huster-wgm/Pytorch-metrics/blob/master/metrics.py
    """

    def __init__(self):
        super(SSIM, self).__init__()

    def guassian(self, w_size, sigma):
        guass = torch.Tensor(
            [
                math.exp(-((x - w_size // 2) ** 2) / float(2 * sigma ** 2))
                for x in range(w_size)
            ]
        )
        return guass / guass.sum()

    def createWindow(self, w_size, channel=1):
        _1D_window = self.guassian(w_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, w_size, w_size).contiguous()
        return window

    def forward(self, y_pred, y_true, w_size=11, size_average=True, full=False):
        """
               args:
                   y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
                   y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
                   w_size : int, default 11
                   size_average : boolean, default True
                   full : boolean, default False
               return ssim, larger the better
               """
        # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
        max_val = torch.max(y_pred)
        min_val = torch.min(y_pred)
        L = max_val - min_val

        padd = 0
        (_, channel, height, width) = y_pred.size()
        window = self.createWindow(w_size, channel=channel).to(y_pred.device)

        mu1 = F.conv2d(y_pred, window, padding=padd, groups=channel)
        mu2 = F.conv2d(y_true, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(y_pred * y_pred, window, padding=padd, groups=channel) - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(y_true * y_true, window, padding=padd, groups=channel) - mu2_sq
        )
        sigma12 = (
            F.conv2d(y_pred * y_true, window, padding=padd, groups=channel) - mu1_mu2
        )

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)

        if full:
            return -ret, cs
        return -ret


class PSNR(nn.Module):
    """PSNR Loss"""

    def __init__(self):
        super(PSNR, self).__init__()

    def forward(self, y_pred, y_true):
        """

        Parameters
        ----------
        y_pred: Tensor
        predicted values

        y_true: Tensor
        ground truth values

        Returns
        -------

        """
        mse = torch.mean((y_pred - y_true) ** 2)
        return 10 * torch.log10(1.0 / mse)


class Column_Difference(nn.Module):
    """This class will calculate the sum of the adjacent column difference between two tensors"""

    def __init__(self):
        super(Column_Difference, self).__init__()

    def column_difference_calculation(self, y_tensor):
        """

        Parameters
        ----------
        y_tensor: Tensor
        The image tensor

        Returns
        -------
        y_new_tensor: Tensor
        The tensor contains the adjacent column difference

        """
        # calculating adjacent column difference
        y_new_tensor = torch.abs(y_tensor[:, 1:, :] - y_tensor[:, :-1, :])
        return y_new_tensor

    def forward(self, y_pred, y_true):
        """

        Parameters
        ----------
        y_pred: Tensor
        predicted values

        y_true: Tensor
        ground truth values

        Returns
        -------
        column_loss: Tensor
        """
        y_pred = self.column_difference_calculation(y_tensor=y_pred)
        y_true = self.column_difference_calculation(y_tensor=y_true)
        column_loss = torch.sum(torch.abs(y_pred - y_true))
        return column_loss


class Row_Difference(nn.Module):
    """This class calculates the sum of the adjacent row difference between two tensors"""

    def __init__(self):
        super(Row_Difference, self).__init__()

    def row_difference_calculation(self, y_tensor):
        """

        Parameters
        ----------
        y_tensor: Tensor
        the image tensor

        Returns
        -------
        y_new_tensor: Tensor
        the tensor contains the adjacent row difference
        """
        y_new_tensor = y_tensor[1:, :, :] - y_tensor[:-1, :, :]
        return y_new_tensor

    def forward(self, y_pred, y_true):
        """

        Parameters
        ----------
        y_pred: Tensor
        The predicted tensor

        y_true: Tensor
        The ground truth tensor

        Returns
        -------
        row_loss
        """
        y_pred = self.row_difference_calculation(y_tensor=y_pred)
        y_true = self.row_difference_calculation(y_tensor=y_true)
        row_loss = torch.sum(torch.abs(y_pred - y_true))
        return row_loss
