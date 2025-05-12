import time
import numpy as np
import torch
import os
import argparse

from models.FCN.fcn import VGGFCN8
from models.HED.hed import HED
from models.U_Net.unet_model import UNet
from models.DeepCrack_Zou.deepcrack import DeepCrack
from models.DeepCrack_Liu.deepcrack_networks import define_deepcrack
from models.FPHBN.fphbn import FPHBN
from models.ECDFFNet.ecdffnet import ECDFFNet
from models.DMA_Net.dmanet import DMANet
from models.AttentionCrackNet.AttentionCrackNet import AttentionCrackNet
from models.CrackFormerII.crackformerII import crackformer
from models.MobileNetV3.segmentation import MobileNetV3Seg
from models.SDDNet.SDDNet import SDDNet
from models.STRNet.STRNet import STRNet

parser = argparse.ArgumentParser()
parser.add_argument('--models_list', type=str, nargs='*',
                    default=['U_Net', 'DeepCrack_Liu', 'FPHBN', 'ECDFFNet', 'DMA_Net', 'AttentionCrackNet', 'DeepCrack_Zou', 'CrackFormer-II', 'MobileNetV3', 'SDDNet', 'STRNet'])
# parser.add_argument('--models_list', type=str, nargs='*',
#                     default=['MobileNetV3', 'SDDNet', 'STRNet', ])
# parser.add_argument('--models_list', type=str, nargs='*',
#                     default=['CrackFormer-II'])
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device('cuda')

models_name_list = args.models_list
for model_name in models_name_list:
    print(model_name, '...')
    if model_name == 'FCN':
        model = VGGFCN8(in_channels=3, n_classes=1)
    elif model_name == 'HED':
        model = HED()
    elif model_name == 'U_Net':
        model = UNet()
    elif model_name == 'DeepCrack_Zou':
        model = DeepCrack()
    elif model_name == 'DeepCrack_Liu':
        model = define_deepcrack(in_nc=3, num_classes=1, ngf=64)
    elif model_name == 'FPHBN':
        model = FPHBN()
    elif model_name == 'ECDFFNet':
        model = ECDFFNet()
    elif model_name == 'DMA_Net':
        model = DMANet()
    elif model_name == 'AttentionCrackNet':
        model = AttentionCrackNet()
    elif model_name == 'CrackFormer-II':
        model = crackformer()
    elif model_name == 'MobileNetV3':
        model = MobileNetV3Seg(1)
    elif model_name == 'SDDNet':
        model = SDDNet()
    elif model_name == 'STRNet':
        model = STRNet()
    else:
        raise NotImplementedError

    model.to(device)
    model.eval()

    x = torch.randn(1, 3, 320, 480).to(device)  # CFD
    # x = torch.randn(1, 3, 640, 352).to(device)  # Crack200
    time_list = []

    for i in range(100):
        start_time = time.time()
        y = model(x)
        end_time = time.time()
        time_list.append(end_time - start_time)

    print(model_name)
    print('average time:', np.mean(time_list), 'average fps:', int(1 / np.mean(time_list)))
    # print('average fps:', int(1 / np.mean(time_list)))
