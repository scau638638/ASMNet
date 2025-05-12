from torch import nn
import os
import torch


class HBCELoss(nn.Module):
    def __init__(self):
        super(HBCELoss, self).__init__()

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"

        mask = (target != 0).float()
        num_positive = torch.sum(mask).float()
        num_negative = mask.numel() - num_positive
        mask[mask != 0] = num_negative / (num_positive + num_negative)
        mask[mask == 0] = num_positive / (num_positive + num_negative)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(predict, target, weight=mask)

        return loss
