# import matplotlib.pyplot as plt
# from tqdm import tqdm#进度条设置
# import matplotlib.pyplot as plt
# # import matplotlib; matplotlib.use('TkAgg')
# from pylab import *
# mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
# mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import torch
from .util.util import reload_population_ckpt, find_train_inds, check_dir, save_population_ckpt, get_gene_len, cxMultiPoint
from tensorboardX import SummaryWriter
from train.train_models_parr import train_population_parr
import numpy as np
import pickle
import os
import random


def bpso(args):
    # gpu_num = len(parser.gpu_id)
    seed = 12
    random.seed(seed)
    np.random.seed(seed)
    optimization_objects = ['f1_score']

    channel = 20
    en_node_num = 4
    de_node_num = 4
    exp_name = 'test'
    epochs = 150  # 130
    batch_size = 4

    devices = [torch.device(type='cuda', index=i) for i in args.gpu_id]
    # devices = [torch.device(type='cuda', index=0)]
    # optimizer_name = 'Lookahead(Adam)'
    optimizer_name = 'Adam'
    learning_rate = 0.001
    l2_weight_decay = 0

    resume_train = False

    train_set_name = 'CFD'
    valid_set_name = 'CFD'

    # --------血管数据集
    # train_set_root = os.path.join(os.path.abspath('.'), 'dataset', 'trainset', train_set_name)
    # valid_set_root = os.path.join(os.path.abspath('.'), 'dataset', 'validset', valid_set_name)

    # --------裂缝数据集
    train_set_root = os.path.join(os.path.abspath('.'), 'dataset', train_set_name)
    valid_set_root = os.path.join(os.path.abspath('.'), 'dataset', valid_set_name)

    func_type = ['conv_relu', 'conv_bn_relu', 'relu_conv', 'bn_relu_conv']

    layer_num_list = [2, 3, 4, 5]

    en_node_num_max_list = [en_node_num for _ in range(max(layer_num_list) + 1)]
    de_node_num_max_list = [de_node_num for _ in range(max(layer_num_list))]

    gene_len = get_gene_len(de_func_type=func_type, en_func_type=func_type, de_node_num_list=de_node_num_max_list,
                            en_node_num_list=en_node_num_max_list, layer_num_list=layer_num_list,
                            de_node_num=de_node_num, en_node_num=en_node_num, only_en=False)

    layer_num_list_len = len(layer_num_list)
    layer_num_gene_len = int(np.ceil(np.log2(layer_num_list_len)))

    model_settings = {'channel': channel, 'en_func_type': func_type, 'de_func_type': func_type,
                      'layer_num_list': layer_num_list, 'layer_num_gene_len': layer_num_gene_len,
                      'en_node_num': en_node_num, 'de_node_num': de_node_num}

    N = 20  # 群体粒子个数 20
    D = gene_len  # 粒子维数
    c1 = 1.5  # 学习因子1
    c2 = 1.5  # 学习因子2
    w = 1  # 惯性因子，一般取1
    V_max = 10  # 速度最大值
    V_min = -10  # 速度最小值
    afa = 10  # 惩罚系数
    G = 50  # 迭代次数

    # 初始化种群
    x = np.random.choice([0, 1], size=(N, D))
    # 初始化速度
    v = np.random.random(size=(N, D))

    check_dir(exp_name)
    sum_writer = SummaryWriter(log_dir=os.path.join(os.path.abspath('.'), 'exps/{}/runs'.format(exp_name)))

    if resume_train:
        g = 0
        exp_name_load = 'test'

        ckpt = reload_population_ckpt(exp_name_load, g)

        pbest = ckpt[0:N]
        p_fitness = ckpt[-1]

        g_fitness = max(p_fitness)
        gbest = pbest[np.array(p_fitness).argmax()]


    else:
        print('==========Sucessfully initialize x and v==========')

        train_list = find_train_inds(x)
        print('gens_{} train individuals is:'.format(0), train_list)

        metrics = train_population_parr(train_list=train_list, gen_num=0, population=x, batch_size=batch_size,
                                        devices=devices, epochs=epochs, exp_name=exp_name,
                                        train_set_name=train_set_name,
                                        valid_set_name=valid_set_name, train_set_root=train_set_root,
                                        valid_set_root=valid_set_root, optimizer_name=optimizer_name,
                                        learning_rate=learning_rate,
                                        model_settings=model_settings, l2_weight_decay=l2_weight_decay)
        print('fitness of all trained model:', metrics)

        p_fitness_all = []
        for i in range(len(x)):
            fitness = []
            for opt_obj in optimization_objects:
                fitness.append(metrics[i][opt_obj])
            p_fitness_all.append(fitness)

        p_fitness = [p_fit[0] for p_fit in p_fitness_all]  # 种群各个粒子的初始适应度值

        # 种群初始最优适应度值
        g_fitness = max(p_fitness)

        print('evaluate gens_{} successfully'.format(0))
        save_population_ckpt(population=x, fitness=p_fitness, exp_name=exp_name, g=0)

        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        g = 0
        sum_writer.add_scalar('best_fitness', g_fitness, g)

        # 初始的个体最优位置和种群最优位置
        pbest = x
        gbest = x[np.array(p_fitness).argmax()]

    for n in range(g + 1, G):
        pbest = pbest.copy()
        p_fitness = p_fitness.copy()
        gbest = gbest.copy()
        # g_fitness = g_fitness.copy()

        # 更新速度
        r1 = np.random.random((N, 1))  # (粒子个数,1)
        r2 = np.random.random((N, 1))
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)  # 直接对照公式写就好了
        # 防止越界处理
        v[v < V_min] = np.random.random() * (V_max - V_min) + V_min
        v[v > V_max] = np.random.random() * (V_max - V_min) + V_min

        # 更新位置
        for i in range(N):  # 遍历每一个粒子
            # 修改速度为sigmoid形式

            v[i, :] = 1. / (1 + np.exp(-np.array(v[i, :])))
            for j in range(D):  # 遍历粒子中的每一个元素
                rand = np.random.random()  # 生成 0-1之间的随机数
                if v[i, j] > rand:
                    x[i, j] = 1
                else:
                    x[i, j] = 0

        print('gens_{} update x and v successfully'.format(n))

        train_list = find_train_inds(x)
        print('gens_{} train individuals is:'.format(n), train_list)
        print('train individuals code are:', x[:])

        metrics = train_population_parr(train_list=train_list, gen_num=n, population=x, batch_size=batch_size,
                                        devices=devices, epochs=epochs, exp_name=exp_name,
                                        train_set_name=train_set_name,
                                        valid_set_name=valid_set_name, train_set_root=train_set_root,
                                        valid_set_root=valid_set_root, optimizer_name=optimizer_name,
                                        learning_rate=learning_rate,
                                        model_settings=model_settings, l2_weight_decay=l2_weight_decay)
        print('fitness of all trained model:', metrics)

        p_fitness_all = []
        for i in range(len(x)):
            fitness = []
            for opt_obj in optimization_objects:
                fitness.append(metrics[i][opt_obj])
            p_fitness_all.append(fitness)

        p_fitness2 = [p_fit[0] for p_fit in p_fitness_all]  # 种群各个粒子的适应度值

        # # 种群最优适应度值
        # g_fitness = max(p_fitness2)

        # 更新每个粒子的历史最优位置
        for i in range(N):
            if p_fitness2[i] > p_fitness[i]:
                pbest[i] = x[i]
                p_fitness[i] = p_fitness2[i]

        # 更新群体的最优位置
        for i in range(N):
            if p_fitness[i] > g_fitness:
                gbest = pbest[i]
                g_fitness = p_fitness[i]

        print('evaluate gens_{} successfully'.format(n))
        save_population_ckpt(population=pbest, fitness=p_fitness, exp_name=exp_name, g=n)

        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        sum_writer.add_scalar('best_fitness', g_fitness, n)

    pickle_file = open(
        os.path.join(os.path.abspath('.'), 'exps/{}/pickle/best_individuals_code.pkl'.format(exp_name)),
        'wb')
    pickle.dump(gbest, pickle_file)
    pickle_file.close()

    print('code are:', gbest[:])



