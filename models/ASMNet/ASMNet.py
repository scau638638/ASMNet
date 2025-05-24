from torch import nn
from models.bo.cbam import CBAM
import torch.nn.functional as F
from models.bo.util import get_gene_cnt_i

class BONet_dif_low(nn.Module):
    def __init__(self, node_gene=None, node_connect=None):
        super(BONet_dif_low, self).__init__()

        # 如果未提供node_gene和node_connect就报错
        if node_gene is None or node_connect is None:
            raise ValueError("必须提供node_gene和node_connect参数")

        self.node_gene = node_gene
        self.node_connect = node_connect

        self.initial_conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.finconv = nn.Conv2d(16, 1, 1, padding=0)
        self.max_pool = nn.MaxPool2d(2, 2)

        self.ConvTran = nn.ModuleList([
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        ])

        self.operation_conv11 = nn.ModuleList([
            nn.Conv2d(16, 32, kernel_size=1, padding=0),
            nn.Conv2d(32, 64, kernel_size=1, padding=0),
            nn.Conv2d(64, 128, kernel_size=1, padding=0),
            nn.Conv2d(128, 256, kernel_size=1, padding=0),
        ])

        # 创建基本操作字典
        self.operations = nn.ModuleDict()

        # 获取所有需要的操作
        required_operations = set()
        for gene_list in self.node_gene:
            for gene in gene_list:
                if gene:  # 确保gene不是None或空字符串
                    required_operations.add(gene)

        # 定义层级尺寸
        channel_sizes = [16, 32, 64, 128, 256]

        # 仅为需要的操作创建模块
        for op_name in required_operations:
            self.operations[op_name] = nn.ModuleList()

            for idx, channels in enumerate(channel_sizes):
                if op_name == '3×3conv':
                    self.operations[op_name].append(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
                elif op_name == 'ReLU':
                    self.operations[op_name].append(nn.ReLU())
                elif op_name == 'BN':
                    self.operations[op_name].append(nn.BatchNorm2d(channels))
                elif op_name == 'BN+3×3conv':
                    self.operations[op_name].append(nn.Sequential(
                        nn.BatchNorm2d(channels),
                        nn.Conv2d(channels, channels, kernel_size=3, padding=1)
                    ))
                elif op_name == 'ReLU+3×3conv':
                    self.operations[op_name].append(nn.Sequential(
                        nn.ReLU(),
                        nn.Conv2d(channels, channels, kernel_size=3, padding=1)
                    ))
                elif op_name == '3×3conv+BN':
                    self.operations[op_name].append(nn.Sequential(
                        nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                        nn.BatchNorm2d(channels)
                    ))
                elif op_name == '3×3conv+ReLU':
                    self.operations[op_name].append(nn.Sequential(
                        nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                        nn.ReLU()
                    ))
                elif op_name == 'BN+ReLU':
                    self.operations[op_name].append(nn.Sequential(
                        nn.BatchNorm2d(channels),
                        nn.ReLU()
                    ))
                elif op_name == 'ReLU+BN':
                    self.operations[op_name].append(nn.Sequential(
                        nn.ReLU(),
                        nn.BatchNorm2d(channels)
                    ))
                elif op_name == 'BN+ReLU+3×3conv':
                    self.operations[op_name].append(nn.Sequential(
                        nn.BatchNorm2d(channels),
                        nn.ReLU(),
                        nn.Conv2d(channels, channels, kernel_size=3, padding=1)
                    ))
                elif op_name == 'BN+3×3conv+ReLU':
                    self.operations[op_name].append(nn.Sequential(
                        nn.BatchNorm2d(channels),
                        nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                        nn.ReLU()
                    ))
                elif op_name == '3×3conv+BN+ReLU':
                    self.operations[op_name].append(nn.Sequential(
                        nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                        nn.BatchNorm2d(channels),
                        nn.ReLU()
                    ))
                elif op_name == 'ReLU+3×3conv+BN':
                    self.operations[op_name].append(nn.Sequential(
                        nn.ReLU(),
                        nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                        nn.BatchNorm2d(channels)
                    ))
                elif op_name == 'ReLU+BN+3×3conv':
                    self.operations[op_name].append(nn.Sequential(
                        nn.ReLU(),
                        nn.BatchNorm2d(channels),
                        nn.Conv2d(channels, channels, kernel_size=3, padding=1)
                    ))
                elif op_name == '3×3conv+ReLU+BN':
                    self.operations[op_name].append(nn.Sequential(
                        nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(channels)
                    ))
                elif op_name == 'cbam':
                    self.operations[op_name].append(CBAM(channels))


    def forward(self, x):
        # 编码
        x1 = self.initial_conv(x)
        enc1 = x1 + self.conv_block(x1, self.node_gene[0], self.node_connect[0], 0)
        x2 = self.operation_conv11[0](self.max_pool(enc1))
        enc2 = x2 + self.conv_block(x2, self.node_gene[1], self.node_connect[1], 1)
        x3 = self.operation_conv11[1](self.max_pool(enc2))
        enc3 = x3 + self.conv_block(x3, self.node_gene[2], self.node_connect[2], 2)
        x4 = self.operation_conv11[2](self.max_pool(enc3))
        enc4 = x4 + self.conv_block(x4, self.node_gene[3], self.node_connect[3], 3)
        x5 = self.operation_conv11[3](self.max_pool(enc4))

        bottle = self.conv_block(x5, self.node_gene[4], self.node_connect[4], 4)

        # d4 = self.ConvTran[0](bottle)
        # dec4 = enc4 + d4 + self.conv_block(d4, self.node_gene[5], self.node_connect[5], 3)
        #
        # d3 = self.ConvTran[1](dec4)
        # dec3 = enc3 + d3 + self.conv_block(d3, self.node_gene[6], self.node_connect[6], 2)
        # d2 = self.ConvTran[2](dec3)
        # dec2 = enc2 + d2 + self.conv_block(d2, self.node_gene[7], self.node_connect[7], 1)
        # d1 = self.ConvTran[3](dec2)
        # dec1 = enc1 + d1 + self.conv_block(d1, self.node_gene[8], self.node_connect[8], 0)

        # Decoder path with size alignment
        d4 = self.ConvTran[0](bottle)
        # 确保d4和enc4尺寸匹配
        if d4.shape[2:] != enc4.shape[2:]:
            d4 = F.interpolate(d4, size=enc4.shape[2:], mode='bilinear', align_corners=True)
        conv_out4 = self.conv_block(d4, self.node_gene[5], self.node_connect[5], 3)
        dec4 = enc4 + d4 + conv_out4

        d3 = self.ConvTran[1](dec4)
        # 确保d3和enc3尺寸匹配
        if d3.shape[2:] != enc3.shape[2:]:
            d3 = F.interpolate(d3, size=enc3.shape[2:], mode='bilinear', align_corners=True)
        conv_out3 = self.conv_block(d3, self.node_gene[6], self.node_connect[6], 2)
        dec3 = enc3 + d3 + conv_out3

        d2 = self.ConvTran[2](dec3)
        # 确保d2和enc2尺寸匹配
        if d2.shape[2:] != enc2.shape[2:]:
            d2 = F.interpolate(d2, size=enc2.shape[2:], mode='bilinear', align_corners=True)
        conv_out2 = self.conv_block(d2, self.node_gene[7], self.node_connect[7], 1)
        dec2 = enc2 + d2 + conv_out2

        d1 = self.ConvTran[3](dec2)
        # 确保d1和enc1尺寸匹配
        if d1.shape[2:] != enc1.shape[2:]:
            d1 = F.interpolate(d1, size=enc1.shape[2:], mode='bilinear', align_corners=True)
        conv_out1 = self.conv_block(d1, self.node_gene[8], self.node_connect[8], 0)
        dec1 = enc1 + d1 + conv_out1

        return self.finconv(dec1)

    def gene(self, x, gene, idx):
        if gene in self.operations:
            return self.operations[gene][idx](x)
        else:
            raise ValueError(f"Unknown gene: {gene}")

    def conv_block(self, x, conv_gene, node_connect, idx):

        out1 = self.gene(x, conv_gene[0], idx)

        out2 = self.gene(out1 if node_connect[0] else x, conv_gene[1], idx)

        out3_inputs = []
        if node_connect[1] == 1:
            out3_inputs.append(out1)
        if node_connect[3] == 1:
            out3_inputs.append(out2)
        if not out3_inputs:
            out3_inputs = [x]
        out3 = self.gene(sum(out3_inputs), conv_gene[3], idx)

        out4_inputs = []
        if node_connect[2] == 1:
            out4_inputs.append(out1)
        if node_connect[4] == 1:
            out4_inputs.append(out2)
        if node_connect[5] == 1:
            out4_inputs.append(out3)
        if not out4_inputs:
            out4_inputs = [x]
        out4 = self.gene(sum(out4_inputs), conv_gene[3], idx)

        out_put = []
        if node_connect == [1, 0, 0, 0, 0, 0]:
            out_put.append(out2)
        if (node_connect[1] == 1 or node_connect[3] == 1) and node_connect[5] == 0:
            out_put.append(out3)
        if node_connect[2] == 1 or node_connect[4] == 1 or node_connect[5] == 1:
            out_put.append(out4)
        return sum(out_put)

