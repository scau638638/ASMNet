from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms import functional as TF
import random


class CrackTree260Dataset(Dataset):
    def __init__(self, dataset_root, txt_name: str = 'train', augment=None):
        super(CrackTree260Dataset, self).__init__()
        assert os.path.exists(dataset_root), f"path '{dataset_root}' does not exist."

        txt_path = os.path.join(dataset_root, txt_name + '.txt')
        assert os.path.exists(txt_path), f"file '{txt_name}' does not exist"

        images_dir = os.path.join(dataset_root, 'image')
        gt_dir = os.path.join(dataset_root, 'gt')

        with open(txt_path, 'r') as fp:
            name_list = [name.strip() for name in fp if len(name.strip()) > 0]

        self.images = [os.path.join(images_dir, name + '.jpg') for name in name_list]
        self.gt = [os.path.join(gt_dir, name + '.png') for name in name_list]

        self.augment = augment

        self.num_return = 2
        self.txt_name = txt_name

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
            if self.txt_name == 'train':
                flag = True
            img, target = self._default_trans(img, target, flag)
        else:
            img, target = self.augment(img, target)

        # img = img.resize((512, 512))
        # target = target.resize((512, 512))
        #
        # img = F.to_tensor(img)
        # target = F.to_tensor(target)

        return img, target

    def __len__(self):
        return len(self.images)
