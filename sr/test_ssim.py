""" This file is for testing ssim loss"""
import numpy as np
import torch
from losses import SSIM


criterion = SSIM()
# case 1 same image
y_pred = []
for i in range(5):
    y_pred.append(np.random.rand(1, 256, 256))

y_pred = torch.tensor(y_pred, dtype=torch.float32)
y_true = y_pred
loss = criterion(y_pred, y_true)
print("same image SSIM loss is {}".format(loss))

# case 2 different image
y_true_diff = []
for i in range(5):
    y_true_diff.append(np.random.rand(1, 256, 256))

y_true_diff = torch.tensor(y_true_diff, dtype=torch.float32)
loss = criterion(y_pred, y_true_diff)
print("different image SSIM loss is {}".format(loss))
