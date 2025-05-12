import torch
import numpy as np


def get_positive_weight(dataset):
    pos_count, neg_count = get_pos_neg_count(dataset)

    pos_weight = neg_count / pos_count

    pos_weight = torch.from_numpy(np.array(pos_weight))

    return pos_weight


def get_pos_neg_count(dataset):
    pos_count = 0
    neg_count = 0

    for data in dataset:
        for _, targets in data:
            pos_count += targets.sum()
            neg_count += (targets.numel() - targets.sum())

    # for _, train_targets in train_dataset:
    #     pos_count += train_targets.sum()
    #     neg_count += (train_targets.numel() - train_targets.sum())
    #
    # for _, val_targets in val_dataset:
    #     pos_count += val_targets.sum()
    #     neg_count += (val_targets.numel() - val_targets.sum())

    pos_count = pos_count.numpy()
    neg_count = neg_count.numpy()

    return pos_count, neg_count

