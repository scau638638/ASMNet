from torch import nn

from dataset.CFD_dataset import CFDDataset
from dataset.DeepCrack537_dataset import DeepCrack537Dataset
from dataset.CrackTree260_dataset import CrackTree260Dataset
from dataset.CrackTree206_dataset import CrackTree206Dataset
from dataset.CRKWH100_dataset import CRKWH100Dataset
from dataset.CrackLS315_dataset import CrackLS315Dataset
from dataset.Crack200_dataset import Crack200Dataset
from dataset.Crack500_dataset import Crack500Dataset
from .get_positive_weight import get_positive_weight
from dataset.DRIVE_dataset import DRIVE_dataset

class WBCELoss(nn.Module):
    def __init__(self, dataset_name, data_path):
        super(WBCELoss, self).__init__()

        if dataset_name == 'CFD':
            train_dataset = CFDDataset(data_path, 'train')
            # val_dataset = CFDDataset(data_path, 'test')

            dataset = (train_dataset,)
        elif dataset_name == 'DeepCrack537':
            train_dataset = DeepCrack537Dataset(data_path, 'train')

            dataset = (train_dataset,)
        elif dataset_name == 'CrackTree260':
            train_dataset = CrackTree260Dataset(data_path, 'train')

            dataset = (train_dataset,)
        elif dataset_name == 'CrackTree206':
            train_dataset = CrackTree206Dataset(data_path, 'train')

            dataset = (train_dataset,)
        elif dataset_name == 'CRKWH100':
            train_dataset = CRKWH100Dataset(data_path, 'train')

            dataset = (train_dataset,)
        elif dataset_name == 'CrackLS315':
            train_dataset = CrackLS315Dataset(data_path, 'train')

            dataset = (train_dataset,)
        elif dataset_name == 'Crack200':
            train_dataset = Crack200Dataset(data_path, 'train')

            dataset = (train_dataset,)
        elif dataset_name == 'Crack500':
            train_dataset = Crack500Dataset(data_path, 'train')

            dataset = (train_dataset,)
        elif dataset_name == 'VOC':
            train_dataset = CrackLS315Dataset(data_path, 'train')
            dataset = (train_dataset,)
        elif dataset_name == 'DRIVE':
            train_dataset = DRIVE_dataset(data_path, 'train')
            dataset = (train_dataset,)
        else:
            raise NotImplementedError

        # self.pos_weight = 0.1 * get_positive_weight(dataset)
        self.pos_weight = get_positive_weight(dataset)

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"

        loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)(predict, target)

        return loss
