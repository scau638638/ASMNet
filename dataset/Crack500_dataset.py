from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms import functional as TF
import random


class Crack500Dataset(Dataset):
    def __init__(self, dataset_root, mode: str = 'train', augment=None):
        super(Crack500Dataset, self).__init__()
        assert os.path.exists(dataset_root), f"path '{dataset_root}' does not exist."

        if mode == 'train':
            file_dir = os.path.join(dataset_root, 'traincrop')
        elif mode == 'val':
            file_dir = os.path.join(dataset_root, 'valcrop')
        else:
            file_dir = os.path.join(dataset_root, 'testcrop')

        file_list = os.listdir(file_dir)

        self.images = []
        self.gt = []
        for i, file_name in enumerate(file_list):
            if i % 2 == 0:
                self.images.append(os.path.join(file_dir, file_name))
            else:
                self.gt.append(os.path.join(file_dir, file_name))
                # if self.images[-1][:-4] != self.gt[-1][:-4]:
                #     print('false')

        self.augment = augment

        self.num_return = 2
        self.mode = mode

    @staticmethod
    def _default_trans(image, annot, train):

        annot = TF.to_grayscale(annot, num_output_channels=1)
        if train:
            if random.random() < 0.5:
                image = TF.hflip(image)
                annot = TF.hflip(annot)
            #
            if random.random() < 0.5:
                image = TF.vflip(image)
                annot = TF.vflip(annot)
            if random.random() < 0.6:
                angle = random.random() * 360
                image = TF.rotate(img=image, angle=angle)
                annot = TF.rotate(img=annot, angle=angle)

        image = TF.to_tensor(image)
        # image = TF.normalize(image, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

        annot = TF.to_tensor(annot)
        # annot[annot > 0.5] = 1
        # annot[annot < 0.5] = 0
        return image, annot

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.gt[index]).convert('1')

        if self.augment is None:
            flag = False
            if self.mode == 'train':
                flag = True
            img, target = self._default_trans(img, target, flag)
        else:
            img, target = self.augment(img, target)

        # img = F.to_tensor(img)
        # target = F.to_tensor(target)

        return img, target

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    data = Crack500Dataset(os.path.join(os.path.abspath('.'), 'Crack500'), 'train')
