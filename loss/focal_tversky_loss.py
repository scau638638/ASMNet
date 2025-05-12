import torch
from torch import nn


class FocalTverskyLoss(nn.Module):
    def __init__(self, smooth=1, alpha=0.5, beta=0.5, gama=1.3):
        super(FocalTverskyLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.gama = gama

    def forward(self, predict, target):
        predict = torch.sigmoid(predict)

        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        TP = torch.sum(predict * target, dim=1)
        FP = torch.sum((1 - target) * predict, dim=1)
        FN = torch.sum(target * (1 - predict), dim=1)

        numerator = TP + self.smooth
        denominator = TP + self.alpha * FP + self.beta * FN + self.smooth

        TL = numerator / denominator

        Focal_Tversky_loss = (1 - TL).pow(self.gama)

        return torch.mean(Focal_Tversky_loss)
