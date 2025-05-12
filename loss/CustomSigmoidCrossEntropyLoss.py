import torch
from torch import nn
import scipy
import numpy as np


# class CustomSigmoidCrossEntropyLossLayer(nn.Module):
#     def __init__(self):
#         super(CustomSigmoidCrossEntropyLossLayer, self).__init__()
#
#     def setup(self, bottom, top):
#         # check for all inputs
#         if len(bottom) == 2:
#             print("Use two inputs (scores and labels) to perform normal sigmoid crossentropy loss")
#         if len(bottom) == 3:
#             print("Use three inputs (scores, labels, weights) to perform ada boost sigmoid crossentropy loss")
#     def reshape(self, bottom, top):
#         # check input dimensions match between the scores and labels
#         if bottom[0].count != bottom[1].count:
#             raise Exception("Inputs must have the same dimension.")
#         # difference would be the same shapeas any input
#         self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
#         # positive and negative sample weights
#         self.pos_neg_w=np.ones(bottom[1].data.shape)
#         if len(bottom) == 3:
#            self.diff_scoreup_label=np.zeros(bottom[1].data.shape)
#         # layer output would be an averaged scalar loss
#         top[0].reshape(1)
#
#     def forward(self, bottom, top):
#         #print('bottom count',bottom[1].count)
#         #print('bottom num', bottom[1].num)
#         score=bottom[0].data
#         label=bottom[1].data
#         #get score from upper layer
#         if len(bottom) == 3:
#             score_up = bottom[2].data
#             #conver score to value in 0-1
#             score_up=1/(1+np.exp(-1*np.absolute(score_up)))
#             #compute the difference between score_up and label
#             self.diff_scoreup_label=np.absolute(score_up-label)
#         first_term=np.maximum(score,0)
#         second_term=-1*score*label
#         third_term=np.log(1+np.exp(-1*np.absolute(score)))
#         # positive and negative sample weights
#         #self.pos_neg_w=np.ones(label.shape)
#         # positive sample number
#         pos_n=(label == 1).sum().sum().astype(float)
#         # negative sample number
#         neg_n = (label == 0).sum().sum().astype(float)
#         #compute positive weights
#         self.pos_neg_w[label==1]=neg_n/(pos_n+neg_n)
#         #compute negative weights
#         self.pos_neg_w[label == 0] = pos_n /(pos_n + neg_n)
#         # the sum of positives and negatives
#         #all_n=pos_n+neg_n
#         #loss at each pixel
#         #print(bottom[0].num)
#         loss_pix=np.multiply(first_term + second_term + third_term,self.pos_neg_w)
#         #if there is score from upper layer, re-weight the samples
#         if len(bottom) == 3:
#             loss_pix=np.multiply(loss_pix,1+self.diff_scoreup_label)
#         top[0].data[...]=np.sum(loss_pix)
#         sig=scipy.special.expit(score)
#         self.diff=(sig-label)
#         if np.isnan(top[0].data):
#                 exit()
#
#     def backward(self, top, propagate_down, bottom):
#         #difference with weights
#         diff_w=np.multiply(self.diff, self.pos_neg_w)
#         if len(bottom) == 3:
#             diff_w =np.multiply(diff_w,1 + self.diff_scoreup_label)
#         bottom[0].diff[...]=diff_w


class CustomSigmoidCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CustomSigmoidCrossEntropyLoss, self).__init__()

    def forward(self, predict, target):

        if len(predict) == 2:
            predict_up = predict[1]
            prob_up = torch.sigmoid(predict_up)
            diff_probup_target = prob_up - target

        mask = (target != 0).float()
        num_positive = torch.sum(mask).float()
        num_negative = mask.numel() - num_positive
        mask[mask != 0] = num_negative / (num_positive + num_negative)
        mask[mask == 0] = num_positive / (num_positive + num_negative)

        loss_pix = torch.nn.functional.binary_cross_entropy_with_logits(predict[0], target, weight=mask, reduction='none')

        if len(predict) == 2:
            loss_pix = loss_pix * (1 + diff_probup_target)

        # loss = torch.sum(loss_pix)
        loss = torch.mean(loss_pix)

        return loss
