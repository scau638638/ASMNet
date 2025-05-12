import torch
from PIL import Image
from torchvision.transforms import functional as F
import numpy as np
import os
import pickle
import argparse
from metrics.calculate_metrics import calculate_metrics
import json

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
from metrics.average_meter import AverageMeter
from torch.nn import functional as FF
# from metrics import prob2binary_maps

def tensor2ndarray(tensor):
    if tensor.requires_grad:
        tensor = tensor.detach()

    if tensor.device != torch.device('cpu'):
        tensor = tensor.cpu()

    return tensor.numpy()


if __name__ == '__main__':

    device = torch.device('cuda:0')

    parser = argparse.ArgumentParser()
    # parser.add_argument('--gpu_id', type=int, nargs='*', default=[0])
    # parser.add_argument('--dataset_list', type=str, nargs='*',
    #                     default=['CFD', 'CrackLS315', 'CrackTree206', 'Crack200',])
    parser.add_argument('--dataset_list', type=str, nargs='*',
                        default=['Crack500'])
    # parser.add_argument('--models_list', type=str, nargs='*',
    #                     default=['U_Net', 'DeepCrack_Liu', 'FPHBN', 'ECDFFNet', 'DMA_Net', 'AttentionCrackNet', 'DeepCrack_Zou', 'CrackFormer-II'])
    parser.add_argument('--models_list', type=str, nargs='*',
                        default=['MobileNetV3', 'SDDNet', 'STRNet', ])
    # parser.add_argument('--models_list', type=str, nargs='*',
    #                     default=['CrackFormer-II'])
    args = parser.parse_args()

    # exp_name = 'test'
    # pickle_file = open(os.path.join(os.path.abspath('.'),
    #                                 'exps/{}/pickle/best_individuals_code.pkl'.format(exp_name)), 'rb')
    # gene = pickle.load(pickle_file)
    # pickle_file.close()
    #
    # channel = 20
    # en_node_num = 4
    # de_node_num = 4
    #
    # epochs = 1000
    # batch_size = 4
    #
    # func_type = ['conv_relu', 'conv_bn_relu', 'relu_conv', 'bn_relu_conv']
    #
    # layer_num_list = [2, 3, 4, 5]
    #
    # layer_num_list_len = len(layer_num_list)
    # layer_num_gene_len = int(np.ceil(np.log2(layer_num_list_len)))
    #
    # model_settings = {'channel': channel, 'en_func_type': func_type, 'de_func_type': func_type,
    #                   'layer_num_list': layer_num_list, 'layer_num_gene_len': layer_num_gene_len,
    #                   'en_node_num': en_node_num, 'de_node_num': de_node_num}

    models_name_list = args.models_list
    models_list = []

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

        # print('model:', model_name)
        for dataset in args.dataset_list:
            model.load_state_dict(torch.load(r'exps/{}-{}/ckpt/{}_best_model.pth'.format(model_name, dataset, dataset)), strict=False)
            model.to(device)
            model.eval()

            path = r'predict\{}\test_image'.format(dataset)
            path_gt = r'predict\{}\test_gt'.format(dataset)
            image_name_list = os.listdir(path)

            # epoch_recall = AverageMeter()
            # epoch_precision = AverageMeter()
            # epoch_f1_score = AverageMeter()
            # epoch_iou = AverageMeter()

            dict_meteics = dict()
            for name in image_name_list:
                image_path = os.path.join(path, name)

                img = Image.open(image_path).convert('RGB')

                img = F.to_tensor(img)

                img = torch.unsqueeze(img, dim=0).to(device)

                output = model(img)

                if model_name in ('HED', 'DeepCrack_Zou', 'FPHBN'):
                    output = torch.sigmoid(output[0])
                elif model_name in ('DeepCrack_Liu', 'ECDFFNet', 'CrackFormer-II'):
                    output = torch.sigmoid(output[-1])
                else:
                    output = torch.sigmoid(output)

                # predict
                # prob_maps = torch.sigmoid(output)
                # prob_maps = output
                # prob_maps = prob_maps.squeeze(0)
                # prob_maps = prob_maps.squeeze(0)
                # prob_maps = tensor2ndarray(prob_maps)
                # prob_maps = prob_maps * 255
                #
                # sss = np.uint8(prob_maps)
                #
                # prob_image = Image.fromarray(np.uint8(prob_maps))
                # prob_image.save(
                #     os.path.join(r'predict\{}\{}\test_prediction'.format(dataset, model_name), 'prediction-' + name[: -4] + '.png'))

                # binary
                gt_path = os.path.join(path_gt, 'target-' + name[: -4] + '.png')
                gt = Image.open(gt_path)
                target_numpy = np.array(gt).astype(np.float32)
                # target_numpy[target_numpy == 255] = 1
                target = torch.from_numpy(target_numpy).to(device).float()
                target = target.unsqueeze(0).unsqueeze(0)
                # output = torch.sigmoid(output)

                (recall, precision, f1_score, iou, threshold) = calculate_metrics(preds=output, targets=target,
                                                                                  device=device, fixed_threshold=False)

                dict_meteics[name[: -4]] = f1_score

            with open('every_pic_metrics/' + dataset + '/' + model_name + "every_pic_metrics.json", "w") as file:
                json.dump(dict_meteics, file)

            #     epoch_recall.update(recall)
            #     epoch_precision.update(precision)
            #     epoch_f1_score.update(f1_score)
            #     epoch_iou.update(iou)
            #
            #     img_numpy = np.array(prob_image)
            #     img_numpy[img_numpy / 255 > threshold] = 255
            #     img_numpy[img_numpy != 255] = 0
            #     img_binary = Image.fromarray(img_numpy)
            #
            #     img_binary.save(os.path.join(r'predict\{}\{}\test_binary'.format(dataset, model_name), 'binary-' + name[: -4] + '.png'))
            #
            #     # color
            #     img_numpy = np.array(img_binary)
            #     img_numpy[img_numpy == 255] = 1
            #     target_numpy = np.array(gt).astype(np.float32)
            #     target_numpy[target_numpy == 255] = 1
            #
            #     img_ten = torch.from_numpy(img_numpy).to(device)
            #     target_ten = torch.from_numpy(target_numpy).to(device).float()
            #
            #     img_tensor = img_ten.unsqueeze(0).unsqueeze(0)
            #     target_tensor = target_ten.unsqueeze(0).unsqueeze(0)
            #
            #     img_neg = -1.0 * (img_tensor - 1.0)
            #
            #     kernel = torch.ones(1, 1, 5, 5).to(device)
            #     target_dilation = FF.conv2d(target_tensor, kernel, stride=1, padding=2)
            #     target_dilation[target_dilation > 0] = 1
            #
            #     true_positive = target_dilation * img_tensor
            #     false_positive = img_tensor - true_positive
            #     false_negative = target_tensor * img_neg
            #
            #     true_positive[true_positive > 0] = 1
            #     false_positive[false_positive > 0] = 2
            #     false_negative[false_negative > 0] = 3
            #
            #     result_tensor = true_positive + false_positive + false_negative
            #
            #     result_tensor = result_tensor.squeeze(0).squeeze(0)
            #     result_numpy = result_tensor.cpu().numpy()
            #
            #     result = Image.fromarray(np.uint8(result_numpy), mode='P')
            #     palette_path = "palette_crack.json"
            #     assert os.path.exists(palette_path), f"palette {palette_path} not found."
            #     with open(palette_path, "rb") as f:
            #         palette_dict = json.load(f)
            #         palette = []
            #         for v in palette_dict.values():
            #             palette += v
            #     result.putpalette(palette)
            #
            #     result.save(os.path.join(r'predict\{}\{}\test_color'.format(dataset, model_name), 'color-' + name[: -4] + '.png'))
            #
            # print(model_name, '-', dataset, 'done!')
            # print('pr:', epoch_precision.val, ' re:', epoch_recall.val, ' f1:', epoch_f1_score.val, ' miou:', epoch_iou.val)





# image_path = r'predict\112.jpg'
# img = Image.open(image_path).convert('RGB')
#
# img = F.to_tensor(img)
#
# img = torch.unsqueeze(img, dim=0).to(device)
#
# output = model(img)
# prob_map = torch.sigmoid(output)
# binary_map = prob2binary_maps(prob_map)
# binary_map = binary_map[:, None, :, :]  # (16, 480, 320) to (16, 1, 480, 320)
# binary_map = torch.from_numpy(binary_map)
# binary_map = torch.squeeze(binary_map, dim=0)
# binary_map = torch.squeeze(binary_map, dim=0)
# binary_numpy = np.array(binary_map)
# binary_numpy = binary_numpy.astype('uint8')
#
# binary_img = Image.fromarray(binary_numpy, mode='P')
# palette_path = "palette_crack.json"
# assert os.path.exists(palette_path), f"palette {palette_path} not found."
# with open(palette_path, "rb") as f:
#     palette_dict = json.load(f)
#     palette = []
#     for v in palette_dict.values():
#         palette += v
#         binary_img.putpalette(palette)
#
# aa = np.array(binary_img)
#
# binary_img.save(r'predict\112_predict.png')
