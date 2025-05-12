import os
import time

import torch
import numpy as np
from models.bo.BO_Net_dif_chan import BONet_dif
from models.bo.BO_Net_dif_low_chan_vo import BONet_dif_low
from models.bo.BO_Net_dif_mid_chan import BONet_dif_mid
from models.bo.BO_Net_dif_lowlow_chan import BONet_dif_lowlow
from models.bo.BO_Net_20_chan import BONet_20
from train.train_model import train_one_model
from evolve.util.util import check_dir
import argparse
import sys
from train.util.util import NoDaemonProcessPool
from models.bo.util import get_gene_cnt, get_gene_cnt_i

sys.path.append('../')


def util_fun(u):
    return train_one_model(u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12], u[13], u[14])


def Configs(model_name):
    if model_name == 'U_Net':
        batch_size = 1
        lr = 0.001
        weight_decay = 1e-4
        momentum = 0.99
        optimizer = 'SGD'
        loss_func = 'BCE'
    elif model_name == 'FCN':
        batch_size = 8
        lr = 0.05
        weight_decay = 5e-4
        momentum = 0.9
        optimizer = 'SGD'
        loss_func = 'BCE'
    elif model_name == 'HED':
        batch_size = 10
        lr = 0.0001
        weight_decay = 0.0002
        momentum = 0.9
        optimizer = 'SGD'
        loss_func = 'HED-BCE'
    elif model_name == 'DeepCrack_Zou':
        batch_size = 1
        lr = 0.001
        weight_decay = 0.0005
        momentum = 0.9
        optimizer = 'SGD'
        loss_func = 'BCE'
    elif model_name == 'DeepCrack_Liu':
        batch_size = 1
        lr = 1e-4
        weight_decay = 2e-4
        momentum = 0.9
        optimizer = 'SGD'
        loss_func = 'WBCE'
    elif model_name == 'FPHBN':
        batch_size = 2
        lr = 0.005
        weight_decay = 0.0002
        momentum = 0.9
        optimizer = 'SGD'
        loss_func = 'CustomSigmoidCrossEntropyLoss'
    elif model_name == 'ECDFFNet':
        batch_size = 4
        lr = 0.001
        weight_decay = 0.0002
        momentum = 0.9
        optimizer = 'Adam'
        loss_func = 'HED-BCE'
    elif model_name == 'DMA_Net':
        batch_size = 4
        lr = 0.001
        weight_decay = 0.00001
        momentum = 0.9
        optimizer = 'Adam'
        loss_func = 'wbce_dice_loss'
    elif model_name == 'AttentionCrackNet':
        batch_size = 4
        lr = 0.001
        weight_decay = 0
        momentum = 0.9
        optimizer = 'Adam'
        loss_func = 'WBCE'
    elif model_name == 'RHACrackNet':
        batch_size = 4
        lr = 0.001
        weight_decay = 0
        momentum = 0.9
        optimizer = 'Adam'
        loss_func = 'WBCE'
    elif model_name == 'CrackFormer-II':
        batch_size = 1
        lr = 0.001
        weight_decay = 0
        momentum = 0.9
        optimizer = 'SGD'
        loss_func = 'cross_entropy_loss_RCF'
    elif model_name == 'MobileNetV3':
        batch_size = 2
        lr = 0.001
        weight_decay = 0.0002
        momentum = 0.9
        optimizer = 'Adam'
        loss_func = 'wbce_dice_loss'
    elif model_name == 'SDDNet':
        batch_size = 2
        lr = 0.001
        weight_decay = 0.00004
        momentum = 0.9
        optimizer = 'Adam'
        loss_func = 'miou_loss'
    elif model_name == 'STRNet':
        batch_size = 2
        lr = 0.001
        weight_decay = 0.00004
        momentum = 0.9
        optimizer = 'Adam'
        loss_func = 'focal_tversky_loss'
    elif model_name == 'BONet':     #2 0.001 0.0004
        batch_size = 4
        lr = 0.001
        weight_decay = 0.0004    #0.0004
        momentum = 0.9
        optimizer = 'Adam'
        loss_func = 'wbce_dice_loss'
    elif model_name == 'OurNet':
        batch_size = 4
        lr = 0.0001
        weight_decay = 0.0002
        momentum = 0.9
        optimizer = 'RAdam'
        loss_func = 'wbce_dice_loss'
    elif model_name == 'genetic_unet':
        batch_size = 4
        lr = 0.0001
        weight_decay = 0.0002
        momentum = 0.9
        optimizer = 'Adam'
        loss_func = 'wbce_dice_loss'
    else:
        raise KeyboardInterrupt

    return {'batch_size': batch_size, 'lr': lr, 'weight_decay': weight_decay, 'momentum': momentum,
            'optimizer': optimizer, 'loss_func': loss_func}


if __name__ == '__main__':

    start_time = time.time()
    from multiprocessing import freeze_support

    freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, nargs='*', default=[0])
    # parser.add_argument('--dataset_list', type=str, nargs='*', default=['CFD', 'CrackTree206', 'CRKWH100', 'CrackLS315', 'DeepCrack537', 'Crack200', 'Crack500'])
    parser.add_argument('--dataset_list', type=str, nargs='*',
                        # default=['CFD'])
                      default=[ 'CFD'])
    # parser.add_argument('--dataset_list', type=str, nargs='*',
    #                     default=['Crack500'])
    # parser.add_argument('--models_list', type=str, nargs='*', default=['U_Net', 'DeepCrack_Liu', 'FPHBN', 'ECDFFNet', 'DMA_Net', 'AttentionCrackNet', 'DeepCrack_Zou', 'OurNet'])
    # parser.add_argument('--models_list', type=str, nargs='*',
    #                     default=['CrackFormer-II','OurNet','BONet'])
    parser.add_argument('--models_list', type=str, nargs='*',
                        default=['BONet'])
    # parser.add_argument('--models_list', type=str, nargs='*',
    #                     default=['CrackFormr-II'])
    args = parser.parse_args()


    devices = [torch.device(type='cuda', index=i) for i in args.gpu_id]
    train_set_name_list = args.dataset_list
    models_name_list = args.models_list

    epochs = 500

    gpu_num = len(devices)

    models_list = []
    configs_list = []

    # for model_name in models_name_list:
        # if model_name == 'FCN':
        #     models_list.append(VGGFCN8(in_channels=3, n_classes=1))
        # elif model_name == 'HED':
        #     models_list.append(HED())
        # elif model_name == 'U_Net':
        #     models_list.append(UNet())
        # elif model_name == 'DeepCrack_Zou':
        #     models_list.append(DeepCrack())
        # elif model_name == 'DeepCrack_Liu':
        #     models_list.append(define_deepcrack(in_nc=3, num_classes=1, ngf=64))
        # elif model_name == 'FPHBN':
        #     models_list.append(FPHBN())
        # elif model_name == 'ECDFFNet':
        #     models_list.append(ECDFFNet())
        # elif model_name == 'DMA_Net':
        #     models_list.append(DMANet())
        # elif model_name == 'AttentionCrackNet':
        #     models_list.append(AttentionCrackNet())
        # elif model_name == 'CrackFormer-II':
        #     models_list.append(crackformer())
        # elif model_name == 'MobileNetV3':
        #     models_list.append(MobileNetV3Seg(1))
        # elif model_name == 'SDDNet':
        #     models_list.append(SDDNet())
        # elif model_name == 'STRNet':
        #     models_list.append(STRNet())
        # elif model_name == 'OurNet':
        #     models_list.append(OctaveSegNet())
    #     elif model_name == 'BONet':
    #         all_node_gene, all_node_connect = get_gene_cnt()
    #         models_list.append(BONet_dif(node_gene=all_node_gene[46], node_connect=all_node_connect[46]))
    #
    #     configs_list.append(Configs(model_name))
    #
    # for dataset in args.dataset_list:
    #     train_set_root = os.path.join(os.path.abspath('.'), 'dataset', dataset)
    #     valid_set_root = os.path.join(os.path.abspath('.'), 'dataset', dataset)
    #
    #     for i in np.arange(0, len(models_name_list), gpu_num):
    #         process_num = np.min((i + gpu_num, len(models_name_list))) - i
    #
    #         save_name_list = [models_name_list[i + j] + '-' + dataset for j in range(process_num)]
    #         for save_name in save_name_list:
    #             check_dir(save_name)
    #
    #         pool = NoDaemonProcessPool(process_num)
    #         train_args = [(configs_list[i + j]['optimizer'], configs_list[i + j]['lr'],
    #                        configs_list[i + j]['weight_decay'], models_list[i + j], models_name_list[i + j],
    #                        configs_list[i + j]['batch_size'], epochs, devices[j],
    #                        dataset, dataset, train_set_root, valid_set_root,
    #                        save_name_list[j], configs_list[i + j]['loss_func'],
    #                        configs_list[i + j]['momentum']) for j in range(process_num)]
    #         pool.map(util_fun, train_args)
    #         pool.terminate()

    with open(r'C:\Users\chenrui\Desktop\BONet-5\models\bo\train_model_num.txt', 'r') as file:
        n = int(file.read())

    all_node_gene, all_node_connect = get_gene_cnt_i(n)
        # 跑一个模型

    from models.bo.test import all_gene,all_con

    with open(r'C:\Users\chenrui\Desktop\BONet-5\models\bo\temp_num.txt', 'r') as file:
        n = int(file.read())
    # n = 46
    print("train {} modle".format(n))
    models_list.append(BONet_dif_low(node_gene=all_node_gene[n], node_connect=all_node_connect[n]))
    configs_list.append(Configs('BONet'))

    for dataset in args.dataset_list:
        train_set_root = os.path.join(os.path.abspath('.'), 'dataset', dataset)
        valid_set_root = os.path.join(os.path.abspath('.'), 'dataset', dataset)

        for i in np.arange(0, len(models_name_list), gpu_num):
            process_num = np.min((i + gpu_num, len(models_name_list))) - i

            save_name_list = [models_name_list[i + j] + '-' + dataset for j in range(process_num)]
            for save_name in save_name_list:
                check_dir(save_name)

            pool = NoDaemonProcessPool(process_num)
            train_args = [(configs_list[i + j]['optimizer'], configs_list[i + j]['lr'],
                            configs_list[i + j]['weight_decay'], models_list[i + j], models_name_list[i + j],
                            configs_list[i + j]['batch_size'], epochs, devices[j],
                            dataset, dataset, train_set_root, valid_set_root,
                            save_name_list[j], configs_list[i + j]['loss_func'],
                            configs_list[i + j]['momentum']) for j in range(process_num)]
            pool.map(util_fun, train_args)
            pool.terminate()
    end_time = time.time()

    print(start_time-end_time)


