from torch import nn
# from metrics import get_statistics_with_pixel
import numpy as np
import torch


class mIoULoss(nn.Module):
    def __init__(self, device='cuda:0'):
        super(mIoULoss, self).__init__()
        self.device = device

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"

        # predicts = torch.sigmoid(predicts)
        #
        # prob_maps_list = []
        # targets_list = []
        # for prob_map, target in zip(predicts, targets):
        #     prob_maps_list.append(prob_map.unsqueeze(0))
        #     targets_list.append(target.unsqueeze(0))
        #
        # statistics = []
        # for pred, gt in zip(prob_maps_list, targets_list):
        #     # pred = pred.astype('uint8')
        #     # gt = gt.astype('uint8')
        #     # calculate each image
        #     statistics.append(get_statistics_with_pixel(pred, gt, threshold=0.5, pixel=2))
        #
        # # 先求全部图片的总tp,fp,fn
        # # get tp, fp, fn
        # tp = np.sum([v[0] for v in statistics])
        # fp = np.sum([v[1] for v in statistics])
        # fn = np.sum([v[2] for v in statistics])
        #
        # iou = tp / (tp + fp + fn)
        #
        # return torch.mean(1-iou)

        predict = torch.sigmoid(predict)

        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        TP = torch.sum(predict * target, dim=1)
        FP = torch.sum((1 - target) * predict, dim=1)
        FN = torch.sum(target * (1 - predict), dim=1)

        iou = TP / (TP + FP + FN)

        return torch.mean(1-iou)
