import numpy
from torch.utils.data import DataLoader

from tqdm import tqdm
import torch
from metrics.calculate_metrics import calculate_metrics
# import shutil
from metrics.average_meter import AverageMeter
import torch.multiprocessing
# from torch.nn.utils.clip_grad import clip_grad_norm_
import os
import sys
import numpy as np
import random
# from torch import nn
from thop import profile

from .util.get_optimizer import get_optimizer, get_loss
from dataset.util.get_datasets import get_datasets
# import multiprocessing as mp
# from ptflops import get_model_complexity_info

from torch.utils.tensorboard import SummaryWriter

from models.CrackFormerII.utils.utils import updateLR

sys.path.append('../')


def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs)**exponent

def train_one_model(optimizer_name, learning_rate, l2_weight_decay, model, model_name, batch_size, epochs, device,
                    train_set_name, valid_set_name,
                    train_set_root, valid_set_root, exp_name, loss_name, momentum):

    seed = 12
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True


    model.to(device)
    if model_name == 'DeepCrack_Zou':
        model.weights_init()

    # loss_func = FocalLossForSigmoid(reduction='mean').to(device)

    # loss_func = BCEDiceLoss(dataset_name=train_set_name, data_path=train_set_root).to(device)
    loss_func = get_loss(loss_name, train_set_name, train_set_root).to(device)

    optimizer = get_optimizer(optimizer_name, filter(lambda p: p.requires_grad, model.parameters()), learning_rate, l2_weight_decay, momentum)

    train_set, num_return = get_datasets(train_set_name, train_set_root, True)
    valid_set, _ = get_datasets(valid_set_name, valid_set_root, False)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(dataset=valid_set, batch_size=1, shuffle=False, num_workers=0)

    best_f1_score = 0
    flag = 0
    count = 0

    valid_epoch = 0  # 80、Crack500是30
    # metrics_name = ['flops', 'param', 'accuracy', 'recall', 'specificity', 'precision', 'f1_score', 'auroc', 'iou']
    metrics_name = ['flops', 'param', 'recall', 'precision', 'f1_score', 'iou']
    metrics = {}
    for metric_name in metrics_name:
        if metric_name == 'flops' or metric_name == 'param':
            metrics.update({metric_name: 100})
        else:
            metrics.update({metric_name: 0})

    writer = SummaryWriter(log_dir=os.path.join(os.path.abspath('.'), 'exps/{}/runs'.format(exp_name)))

    if model_name == 'DMA_Net':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=500)
    if model_name == 'AttentionCrackNet':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=300)
    if model_name == 'SDDNet':
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer,
                                                         base_lr=0.00001,
                                                         max_lr=learning_rate,
                                                         step_size_up=2000,
                                                         gamma=0.99996,
                                                         cycle_momentum=False)
    if model_name == 'OurNet':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                                  mode='min',
                                                                  factor=0.9,
                                                                  patience=10,
                                                                  verbose=False,
                                                                  threshold=1e-4,
                                                                  threshold_mode='rel',
                                                                  cooldown=0,
                                                                  min_lr=1e-5,
                                                                  eps=1e-8)
    if model_name == 'BONet':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                                  mode='min',
                                                                  factor=0.9,
                                                                  patience=10,
                                                                  verbose=False,
                                                                  threshold=1e-4,
                                                                  threshold_mode='rel',
                                                                  cooldown=0,
                                                                  min_lr=1e-5,
                                                                  eps=1e-8)
    try:
        if model_name in ('HED', 'DeepCrack_Zou', 'DeepCrack_Liu', 'FPHBN'):
            it = 0

        for i in range(epochs):
            model.train()
            if model_name == 'ECDFFNet':
                if i % 50 == 0:
                    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 10

            if model_name == 'CrackFormer-II':
                optimizer.param_groups[0]['lr'] = updateLR(learning_rate, i, 500)

            if model_name == 'MobileNetV3':
                optimizer.param_groups[0]['lr'] = poly_lr(i, epochs, learning_rate, 0.9)

            if model_name == 'STRNet':
                if i == 30 or i == 70 or i == 120:
                    lr = optimizer.param_groups[0]['lr']
                    optimizer.param_groups[0]['lr'] = lr * 0.8

            train_tqdm_batch = tqdm(iterable=train_loader, total=numpy.ceil(len(train_set) / batch_size))

            for images, targets in train_tqdm_batch:
                if model_name == 'HED':
                    it = it + 1
                    if it == 5000:
                        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 10
                if model_name in ('DeepCrack_Zou', 'FPHBN'):
                    it = it + 1
                    if it == 10000:
                        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 10
                if model_name == 'DeepCrack_Liu':
                    it = it + 1
                    if it == 5e4:
                        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 5

                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad()
                preds = model(images)
                if model_name in ('HED', 'DeepCrack_Zou', 'ECDFFNet'):
                    loss = torch.zeros(1).to(device)
                    for pred in preds:
                        loss += loss_func(pred, targets)
                elif model_name == 'DeepCrack_Liu':
                    loss_side = 0.0
                    for out, w in zip(preds[:-1], [1.0, 1, 1.0, 1.0, 1.0]):
                        loss_side += loss_func(out, targets) * w
                    loss_fused = loss_func(preds[-1], targets)
                    loss = loss_side * 1.0 + loss_fused * 1.0
                elif model_name == 'FPHBN':
                    loss_side = torch.zeros(1).to(device)
                    loss_fuse = torch.zeros(1).to(device)
                    for j in range(len(preds)):
                        if j == 0:
                            loss_fuse = loss_func((preds[0],), targets)
                        elif j == 1:
                            loss_side = loss_func((preds[j],), targets)
                        else:
                            loss_side += loss_func((preds[j], preds[j - 1]), targets)
                    loss = loss_fuse + loss_side
                elif model_name == 'CrackFormer-II':
                    loss = loss_func(preds[-1], targets)
                else:
                    loss = loss_func(preds, targets)
                loss.backward()
                # clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()

            train_tqdm_batch.close()

            print('{}-{}: epoch_{} train end'.format(model_name, train_set_name, i))

            if model_name == 'DMA_Net' or model_name == 'SDDNet':
                lr_scheduler.step()

            # epoch_acc = AverageMeter()
            epoch_recall = AverageMeter()
            epoch_precision = AverageMeter()
            # epoch_specificity = AverageMeter()
            epoch_f1_score = AverageMeter()
            epoch_iou = AverageMeter()
            # epoch_auroc = AverageMeter()

            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], i)

            if (i >= valid_epoch):
                with torch.no_grad():
                    model.eval()
                    valid_tqdm_batch = tqdm(iterable=valid_loader, total=numpy.ceil(len(valid_set) / 1))

                    for images, targets in valid_tqdm_batch:
                        images = images.to(device)
                        targets = targets.to(device)
                        preds = model(images)
                        if model_name in ('HED', 'DeepCrack_Zou', 'FPHBN'):
                            preds = torch.sigmoid(preds[0])
                        elif model_name in ('DeepCrack_Liu', 'ECDFFNet', 'CrackFormer-II'):
                            preds = torch.sigmoid(preds[-1])
                        else:
                            preds = torch.sigmoid(preds)

                        # (acc, recall, specificity, precision,
                        #  f1_score, iou, auroc) = calculate_metrics(preds=preds, targets=targets, device=device)
                        (recall, precision,
                         f1_score, iou, threshold) = calculate_metrics(preds=preds, targets=targets, device=device, fixed_threshold=False)
                        # epoch_acc.update(acc)
                        epoch_recall.update(recall)
                        epoch_precision.update(precision)
                        # epoch_specificity.update(specificity)
                        epoch_f1_score.update(f1_score)
                        epoch_iou.update(iou)
                        # epoch_auroc.update(auroc)

                    if i == valid_epoch:
                        if model_name == 'ECDFFNet':
                            flops = 0.0
                            param = 0.0
                        else:
                             flops, param = profile(model=model, inputs=(images,), verbose=False)
                             flops = flops / 1e11
                             param = param / 1e6
                            # flops = 0.0
                            # param = 0.0

                    true_f1_score = (2*epoch_recall.val*epoch_precision.val)/(epoch_recall.val+epoch_precision.val)

                    writer.add_scalar(metrics_name[0], flops, i)
                    writer.add_scalar(metrics_name[1], param, i)
                    writer.add_scalar(metrics_name[2], epoch_recall.val, i)
                    writer.add_scalar(metrics_name[3], epoch_precision.val, i)
                    writer.add_scalar(metrics_name[4], true_f1_score, i)
                    writer.add_scalar(metrics_name[5], epoch_iou.val, i)


                    print('{}-{}: epoch_{} validate end'.format(model_name, train_set_name, i))
                    # print('acc:{} | recall:{} | spe:{} | pre:{} | f1_score:{} | auroc:{}'
                    #       .format(epoch_acc.val,
                    #               epoch_recall.val,
                    #               epoch_specificity.val,
                    #               epoch_precision.val,
                    #               epoch_f1_score.val,
                    #               epoch_auroc.val))
                    print('recall:{} | pre:{} | f1_score:{} | iou:{}'
                          .format(epoch_recall.val,
                                  epoch_precision.val,
                                  true_f1_score,
                                  epoch_iou.val))
                    if true_f1_score > best_f1_score:
                        best_f1_score = true_f1_score

                        flag = i
                        count = 0
                        for key in list(metrics):
                            if key == 'flops':
                                metrics[key] = flops
                            elif key == 'param':
                                metrics[key] = param
                            # elif key == 'accuracy':
                            #     metrics[key] = epoch_acc.val
                            elif key == 'recall':
                                metrics[key] = epoch_recall.val
                            # elif key == 'specificity':
                            #     metrics[key] = epoch_specificity.val
                            elif key == 'precision':
                                metrics[key] = epoch_precision.val
                            elif key == 'f1_score':
                                metrics[key] = true_f1_score
                            # elif key == 'auroc':
                            #     metrics[key] = epoch_auroc.val
                            elif key == 'iou':
                                metrics[key] = epoch_iou.val
                            else:
                                raise NotImplementedError

                        import pandas as pd
                        from os.path import join
                        performance_df = pd.DataFrame(
                            # data=[[gen_num, ind_num, epoch_acc.val, epoch_recall.val, epoch_specificity.val,
                            #        epoch_precision.val,
                            #        2 * epoch_recall.val * epoch_precision.val / (epoch_recall.val + epoch_precision.val,
                            #        epoch_iou.val, epoch_auroc.val]],
                            # columns=['epoch', 'individual', 'acc', 'recall',
                            #          'specificity', 'precision', 'f1_score', 'iou',
                            #          'auroc', ]
                            data=[[i, epoch_recall.val,
                                   epoch_precision.val,
                                   true_f1_score,
                                   epoch_iou.val, flops, param]],
                            columns=['epoch', 'recall',
                                     'precision', 'f1_score', 'iou', 'flops', 'param']

                        )
                        # performance_csv_path = join(os.path.abspath('.'), 'exps/{}/csv'.format(exp_name),
                        #                             'performance.csv')

                        #BONet
                        with open(r'C:\Users\chenrui\Desktop\BONet-5\models\bo\train_model_num.txt', 'r') as file:
                            model_num = int(file.read())
                        with open(r'C:\Users\chenrui\Desktop\BONet-5\models\bo\temp_num.txt', 'r') as file:
                            temp_num = int(file.read())
                        performance_csv_path = join(os.path.abspath('.'), 'exps/{}/csv'.format(exp_name),
                                                         'performance_{}_{}.csv'.format(model_num, temp_num))
                        performance_df.to_csv(performance_csv_path)

                        torch.save(model.state_dict(), os.path.join('exps/{}/ckpt'.format(exp_name), train_set_name + '_best_model.pth_{}_{}'.format(model_num, temp_num)))

                    else:
                        if i >= valid_epoch:
                            count += 1

                    end = None
                    if i > valid_epoch + 50 and best_f1_score < 0.50:
                        end = True
                    if (count >= 500) or end:
                        print('{}-{}: current best epoch_{} best_f1_score:'.format(model_name, train_set_name, flag), best_f1_score)
                        print('{}-{}: train early stop'.format(model_name, train_set_name))
                        print('=======================================================================')
                        valid_tqdm_batch.close()
                        return metrics, True
                    print('{}-{}: current best epoch_{} best_f1_score:'.format(model_name, train_set_name, flag), best_f1_score)
                    valid_tqdm_batch.close()
        print('{}-{}: best epoch_{} best_f1_score:'.format(model_name, train_set_name, flag), best_f1_score)
        print('=======================================================================')
    except RuntimeError as exception:
        images.detach_()
        del images
        del model
        del targets
        print(exception)
        return metrics, False

    # for i in range(epochs):
    #     train_tqdm_batch = tqdm(iterable=train_loader, total=numpy.ceil(len(train_set) / batch_size))
    #
    #     for images, targets in train_tqdm_batch:
    #         images, targets = images.to(device), targets.to(device)
    #         optimizer.zero_grad()
    #         preds = model(images)
    #         loss = loss_func(preds, targets)
    #         loss.backward()
    #         clip_grad_norm_(model.parameters(), 0.1)
    #         optimizer.step()
    #
    #     train_tqdm_batch.close()
    #
    #     print('gens_{} individual_{}_epoch_{} train end'.format(gen_num, ind_num, i))
    #
    #     # epoch_acc = AverageMeter()
    #     epoch_recall = AverageMeter()
    #     epoch_precision = AverageMeter()
    #     # epoch_specificity = AverageMeter()
    #     epoch_f1_score = AverageMeter()
    #     epoch_iou = AverageMeter()
    #     # epoch_auroc = AverageMeter()
    #
    #     if (i >= valid_epoch):
    #         with torch.no_grad():
    #             model.eval()
    #             valid_tqdm_batch = tqdm(iterable=valid_loader, total=numpy.ceil(len(valid_set) / 1))
    #
    #             for images, targets in valid_tqdm_batch:
    #                 images = images.to(device)
    #                 targets = targets.to(device)
    #                 preds = model(images)
    #
    #                 # (acc, recall, specificity, precision,
    #                 #  f1_score, iou, auroc) = calculate_metrics(preds=preds, targets=targets, device=device)
    #                 (recall, precision,
    #                  f1_score, iou) = calculate_metrics(preds=preds, targets=targets, device=device)
    #                 # epoch_acc.update(acc)
    #                 epoch_recall.update(recall)
    #                 epoch_precision.update(precision)
    #                 # epoch_specificity.update(specificity)
    #                 epoch_f1_score.update(f1_score)
    #                 epoch_iou.update(iou)
    #                 # epoch_auroc.update(auroc)
    #
    #             if i == valid_epoch:
    #                 flops, param = profile(model=model, inputs=(images,), verbose=False)
    #                 flops = flops / 1e11
    #                 param = param / 1e6
    #
    #             print('gens_{} individual_{}_epoch_{} validate end'.format(gen_num, ind_num, i))
    #             # print('acc:{} | recall:{} | spe:{} | pre:{} | f1_score:{} | auroc:{}'
    #             #       .format(epoch_acc.val,
    #             #               epoch_recall.val,
    #             #               epoch_specificity.val,
    #             #               epoch_precision.val,
    #             #               epoch_f1_score.val,
    #             #               epoch_auroc.val))
    #             print('recall:{} | pre:{} | f1_score:{} | iou:{}'
    #                   .format(epoch_recall.val,
    #                           epoch_precision.val,
    #                           epoch_f1_score.val,
    #                           epoch_iou.val))
    #             if epoch_f1_score.val > best_f1_score:
    #                 best_f1_score = epoch_f1_score.val
    #
    #                 flag = i
    #                 count = 0
    #                 for key in list(metrics):
    #                     if key == 'flops':
    #                         metrics[key] = flops
    #                     elif key == 'param':
    #                         metrics[key] = param
    #                     # elif key == 'accuracy':
    #                     #     metrics[key] = epoch_acc.val
    #                     elif key == 'recall':
    #                         metrics[key] = epoch_recall.val
    #                     # elif key == 'specificity':
    #                     #     metrics[key] = epoch_specificity.val
    #                     elif key == 'precision':
    #                         metrics[key] = epoch_precision.val
    #                     elif key == 'f1_score':
    #                         metrics[key] = epoch_f1_score.val
    #                     # elif key == 'auroc':
    #                     #     metrics[key] = epoch_auroc.val
    #                     elif key == 'iou':
    #                         metrics[key] = epoch_iou.val
    #                     else:
    #                         raise NotImplementedError
    #
    #                 import pandas as pd
    #                 from os.path import join
    #                 performance_df = pd.DataFrame(
    #                     # data=[[gen_num, ind_num, epoch_acc.val, epoch_recall.val, epoch_specificity.val,
    #                     #        epoch_precision.val,
    #                     #        epoch_f1_score.val, epoch_iou.val, epoch_auroc.val]],
    #                     # columns=['epoch', 'individual', 'acc', 'recall',
    #                     #          'specificity', 'precision', 'f1_score', 'iou',
    #                     #          'auroc', ]
    #                     data=[[gen_num, ind_num, epoch_recall.val,
    #                            epoch_precision.val,
    #                            epoch_f1_score.val, epoch_iou.val]],
    #                     columns=['epoch', 'individual', 'recall',
    #                              'precision', 'f1_score', 'iou', ]
    #
    #                 )
    #                 performance_csv_path = join(os.path.abspath('.'), 'exps/{}/csv'.format(exp_name),
    #                                             'gens_{} individual_{} performance.csv'.format(gen_num, ind_num))
    #                 performance_df.to_csv(performance_csv_path)
    #             else:
    #                 if i >= valid_epoch:
    #                     count += 1
    #
    #             end = None
    #             if i > valid_epoch + 15 and best_f1_score < 0.50:
    #                 end = True
    #             if (count >= 70) or end:
    #                 print('current best epoch_{} best_f1_score:'.format(flag), best_f1_score)
    #                 print('gens_{} individual_{} train early stop'.format(gen_num, ind_num))
    #                 print('=======================================================================')
    #                 valid_tqdm_batch.close()
    #                 return metrics, True
    #             print('current best epoch_{} best_f1_score:'.format(flag), best_f1_score)
    #             valid_tqdm_batch.close()
    # print('current best epoch_{} best_f1_score:'.format(flag), best_f1_score)
    # print('=======================================================================')

    return metrics, True
