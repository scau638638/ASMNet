from torch import nn
from models.bo.cbam import CBAM
class BONet(nn.Module):
    def __init__(self, in_channels=3, out_channels=20, node_gene=None, node_connect=None):
        super(BONet, self).__init__()

        self.conv_gene = [
            "cbam",
            "3×3conv",
            "BN+3×3conv",
            "ReLU+3×3conv",
            "3×3conv+BN",
            "3×3conv+ReLU",
            "BN+ReLU+3×3conv",
            "BN+3×3conv+ReLU",
            "3×3conv+BN+ReLU",
            "3×3conv+ReLU+BN"
        ]
        self.node_gene = node_gene
        self.node_connect = node_connect

        # 将 3 通道输入直接转换为 20 通道
        self.initial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.initial_norm = nn.BatchNorm2d(out_channels)

        self.conv11 = nn.Conv2d(out_channels, out_channels, 1, padding=1)
        self.finconv = nn.Conv2d(out_channels, 1, 1, padding=0)

        self.operation_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.operation_BNConv = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.operation_ReLUConv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.operation_ConvBN = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.operation_ConvReLU = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.operation_BNReLUConv = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.operation_BNConvReLU = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.operation_ConvBNReLU = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.operation_ConvReLUBN = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

        self.operation_upConv = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.operation_upBNConv = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.operation_upReLUConv = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.operation_upConvBN = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.operation_upConvReLU = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.operation_upBNReLUConv = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.operation_upBNConvReLU = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.operation_upConvBNReLU = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.operation_upConvReLUBN = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

        self.operation_cba = nn.Sequential(
            CBAM(out_channels)
        )

    def forward(self, x, node_gene=None, conv_gene=None):
        # 编码
        # 直接将输入从 3 通道转换为 20 通道
        x = self.initial_conv(x)
        enc1 = x.clone() + self.conv_block(x, self.node_gene[0], self.node_connect[0])
        enc2 = enc1.clone() + self.conv_block(enc1, self.node_gene[1], self.node_connect[1])
        enc3 = enc2.clone() + self.conv_block(enc2, self.node_gene[2], self.node_connect[2])
        enc4 = enc3.clone() + self.conv_block(enc3, self.node_gene[3], self.node_connect[3])
        enc5 = enc4.clone() + self.conv_block(enc4, self.node_gene[4], self.node_connect[4])

        bottle = self.conv_block(enc5, self.node_gene[5], self.node_connect[5])

        # 解码
        dec5 = enc5.clone() + bottle + self.upconv_block(bottle, self.node_gene[6], self.node_connect[6])
        dec4 = enc4.clone() + dec5 + self.upconv_block(dec5, self.node_gene[7], self.node_connect[7])
        dec3 = enc3.clone() + dec4 + self.upconv_block(dec4, self.node_gene[8], self.node_connect[8])
        dec2 = enc2.clone() + dec3 + self.upconv_block(dec3, self.node_gene[9], self.node_connect[9])
        dec1 = enc1.clone() + dec2 + self.upconv_block(dec2, self.node_gene[10], self.node_connect[10])

        return self.finconv(dec1)

    def gene(self, x, gene):
        if gene == '3×3conv':
            return self.operation_conv(x)
        elif gene == 'BN+3×3conv':
            return self.operation_BNConv(x)
        elif gene == 'ReLU+3×3conv':
            return self.operation_ReLUConv(x)
        elif gene == '3×3conv+BN':
            return self.operation_ConvBN(x)
        elif gene == '3×3conv+ReLU':
            return self.operation_ConvReLU(x)
        elif gene == 'BN+ReLU+3×3conv':
            return self.operation_BNReLUConv(x)
        elif gene == 'BN+3×3conv+ReLU':
            return self.operation_BNConvReLU(x)
        elif gene == '3×3conv+BN+ReLU':
            return self.operation_ConvBNReLU(x)
        elif gene == '3×3conv+ReLU+BN':
            return self.operation_ConvReLUBN(x)
        elif gene == 'cbam':
            return self.operation_cba(x)
        else:
            raise ValueError(f"Unknown gene: {gene}")

    def upgene(self, x, gene):
        if gene == "3×3conv":
            return self.operation_upConv(x)
        elif gene == "BN+3×3conv":
            return self.operation_upBNConv(x)
        elif gene == "ReLU+3×3conv":
            return self.operation_upReLUConv(x)
        elif gene == '3×3conv+ReLU':
            return self.operation_upConvReLU(x)
        elif gene == '3×3conv+BN':
            return self.operation_upConvBN(x)
        elif gene == 'BN+ReLU+3×3conv':
            return self.operation_upBNReLUConv(x)
        elif gene == 'BN+3×3conv+ReLU':
            return self.operation_upBNConvReLU(x)
        elif gene == '3×3conv+BN+ReLU':
            return self.operation_upConvBNReLU(x)
        elif gene == '3×3conv+ReLU+BN':
            return self.operation_upConvReLUBN(x)
        elif gene == "cbam":
            return self.operation_cba(x)
        else:
            raise ValueError(f"Unknown gene: {gene}")

    def conv_block(self, x, conv_gene, node_connect):

        out1 = self.gene(x, conv_gene[0])

        out2 = self.gene(out1 if node_connect[0] else x, conv_gene[1])

        out3_inputs = []
        if node_connect[1] == 1:
            out3_inputs.append(out1)
        if node_connect[3] == 1:
            out3_inputs.append(out2)
        if not out3_inputs:
            out3_inputs = [x]
        out3 = self.gene(sum(out3_inputs), conv_gene[3])

        out4_inputs = []
        if node_connect[2] == 1:
            out4_inputs.append(out1)
        if node_connect[4] == 1:
            out4_inputs.append(out2)
        if node_connect[5] == 1:
            out4_inputs.append(out3)
        if not out4_inputs:
            out4_inputs = [x]
        out4 = self.gene(sum(out4_inputs), conv_gene[3])

        out_put = []
        if node_connect == [1, 0, 0, 0, 0, 0]:
            out_put.append(out2)
        if (node_connect[1] == 1 or node_connect[3] == 1) and node_connect[5] == 0:
            out_put.append(out3)
        if node_connect[2] == 1 or node_connect[4] == 1 or node_connect[5] == 1:
            out_put.append(out4)
        return sum(out_put)

    def upconv_block(self, x, conv_gene, node_connect):
        out1 = self.upgene(x, conv_gene[0])

        out2 = self.upgene(out1 if node_connect[0] else x, conv_gene[1])

        out3_inputs = []
        if node_connect[1] == 1:
            out3_inputs.append(out1)
        if node_connect[3] == 1:
            out3_inputs.append(out2)
        if not out3_inputs:
            out3_inputs = [x]
        out3 = self.upgene(sum(out3_inputs), conv_gene[2])

        out4_inputs = []
        if node_connect[2] == 1:
            out4_inputs.append(out1)
        if node_connect[4] == 1:
            out4_inputs.append(out2)
        if node_connect[5] == 1:
            out4_inputs.append(out3)
        if not out4_inputs:
            out4_inputs = [x]
        out4 = self.upgene(sum(out4_inputs), conv_gene[3])

        out_put = []
        if node_connect == [1, 0, 0, 0, 0, 0]:
            out_put.append(out2)
        if (node_connect[1] == 1 or node_connect[3] == 1) and node_connect[5] == 0:
            out_put.append(out3)
        if node_connect[2] == 1 or node_connect[4] == 1 or node_connect[5] == 1:
            out_put.append(out4)
        return sum(out_put)