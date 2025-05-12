from torch import nn
import torch

from .wbce import WBCELoss
from .dice import BinaryDiceLoss


class BCEDiceLoss(nn.Module):
    def __init__(self, dataset_name, data_path, smooth=1e-5, gamma=0.5):
        super(BCEDiceLoss, self).__init__()
        self.smooth = smooth
        self.gamma = gamma

        self.BCE = WBCELoss(dataset_name, data_path)
        self.dice = BinaryDiceLoss()

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"

        # BCE
        BCE_loss = self.BCE(predict, target)

        # dice
        # predict = torch.sigmoid(predict)
        #
        # predict = predict.contiguous().view(predict.shape[0], -1)
        # target = target.contiguous().view(target.shape[0], -1)
        #
        # num = 2 * torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        # den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + self.smooth
        # # den = torch.sum(predict + target, dim=1) + self.smooth
        #
        # dice_loss = torch.mean(1 - num / den)
        #
        # # loss = (1 - self.gamma) * dice_loss + self.gamma * BCE_loss

        dice_loss = self.dice(predict, target)

        loss = dice_loss + BCE_loss

        return loss
