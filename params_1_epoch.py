import os
import time

import torch
import numpy as np
from models.bo.BO_Net_dif_low_chan_vo import BONet_dif_low
from train.train_model_par import train_one_model
from evolve.util.util import check_dir
import argparse
import sys
from train.util.util import NoDaemonProcessPool
from models.bo.util import get_gene_cnt, get_gene_cnt_i

sys.path.append('../')

def util_fun(u):
    return train_one_model(u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12], u[13], u[14])

def Configs():
    batch_size = 4
    lr = 0.001
    weight_decay = 0.0004
    momentum = 0.9
    optimizer = 'Adam'
    loss_func = 'wbce_dice_loss'
    return {'batch_size': batch_size, 'lr': lr, 'weight_decay': weight_decay, 'momentum': momentum,
            'optimizer': optimizer, 'loss_func': loss_func}

def par(gene, connect):
    from multiprocessing import freeze_support

    freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, nargs='*', default=[0])
    parser.add_argument('--dataset_list', type=str, nargs='*',
                        default=['CFD'])
    parser.add_argument('--models_list', type=str, nargs='*',
                        default=['BONet'])
    args = parser.parse_args()

    devices = [torch.device(type='cuda', index=i) for i in args.gpu_id]
    models_name_list = args.models_list

    epochs = 1
    gpu_num = len(devices)

    models_list = []
    configs_list = []

    models_list.append(BONet_dif_low(node_gene=gene, node_connect=connect))
    configs_list.append(Configs())

    for dataset in args.dataset_list:
        train_set_root = r'C:\Users\chenrui\Desktop\BONet-5\dataset\CFD'
        valid_set_root = r'C:\Users\chenrui\Desktop\BONet-5\dataset\CFD'

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


if __name__ == '__main__':
    print()