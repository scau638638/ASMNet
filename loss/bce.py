from torch import nn


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"

        loss = nn.BCEWithLogitsLoss()(predict, target)

        return loss
