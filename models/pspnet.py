import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from models.modules import SynchronizedBatchNorm2d as SyncBatchNorm2d
from models.resnet import resnet50, resnet101


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_dim, pool_sizes):
        super(PyramidPoolingModule, self).__init__()
        out_dim = int(in_dim / len(pool_sizes))
        self.features = []
        for s in pool_sizes:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
                SyncBatchNorm2d(out_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.upsample(f(x), x_size[2:], mode='bilinear'))
        out = torch.cat(out, 1)
        return out


class FPNModule(nn.Module):
    """ This implementation of Feature Pyramid Network is adapted from
        https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/models/models.py
    """

    def __init__(self, num_classes, fpn_dim, in_dims=(256, 512, 1024, 2048)):
        super(FPNModule, self).__init__()
        self.fpn_in = []
        for fpn_in_dim in in_dims:
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_in_dim, fpn_dim, kernel_size=1, bias=False),
                # nn.BatchNorm2d(fpn_dim),
                # nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(in_dims)):
            self.fpn_out.append(nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1),
                # nn.BatchNorm2d(fpn_dim),
                # nn.ReLU(inplace=True)
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.final_conv = nn.Sequential(
            nn.Conv2d(len(in_dims) * fpn_dim, fpn_dim,
                      padding=1, kernel_size=3),
            SyncBatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_dim, num_classes, kernel_size=1)
        )

    def forward(self, down_features):
        last_stage = self.fpn_in[-1](down_features[-1])
        fpn_feature_list = [self.fpn_out[-1](last_stage)]
        for i in reversed(range(len(down_features) - 1)):
            x = self.fpn_in[i](down_features[i])  # bottom-up branch
            last_stage = F.upsample(last_stage, x.size()[2:],
                                    mode='bilinear', align_corners=False)  # upsample
            last_stage = x + last_stage  # feature fusion
            fpn_feature_list.append(
                self.fpn_out[i](last_stage))  # fine feature

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.upsample(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        fpn_feature = self.final_conv(fusion_out)
        return fpn_feature


class PSPFPNet(nn.Module):
    def __init__(self, num_classes, layer,
                 fpn_dim=256, pretrained=False, pool_sizes=(1, 2, 3, 6)):
        super(PSPFPNet, self).__init__()
        if layer not in [18, 34, 50, 101, 152]:
            raise ValueError('Resnet-{} is not supported \n'
                             'Currently supported models are Resnet 18, 34, 50,\n'
                             '101, 152'.format(layer))
        print('Model: PSPFPNet_Resnet{}, Pretrained: {}'.format(
            layer, pretrained))
        if layer == 18:
            self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        elif layer == 34:
            self.resnet = torchvision.models.resnet34(pretrained=pretrained)
        elif layer == 50:
            self.resnet = torchvision.models.resnet50(pretrained=pretrained)
        elif layer == 101:
            self.resnet = torchvision.models.resnet101(pretrained=pretrained)
        elif layer == 152:
            self.resnet = torchvision.models.resnet152(pretrained=pretrained)

        expansion = self.resnet.layer1[0].expansion  # 4
        in_dims = [64, 128, 256, 512]
        in_dims = [dim * expansion for dim in in_dims]
        ppm_indim = in_dims[-1]
        self.pool_sizes = pool_sizes
        self.ppm = PyramidPoolingModule(ppm_indim, self.pool_sizes)
        ppm_outdim = ppm_indim * 2
        in_dims[-1] = ppm_outdim
        self.fpn_module = FPNModule(num_classes, fpn_dim, in_dims=in_dims)

    def forward(self, x):
        input_size = x.size()[2:]
        down_features = []
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)  # 4
        down_features.append(x)
        x = self.resnet.layer2(x)  # 8
        down_features.append(x)
        x = self.resnet.layer3(x)  # 16
        down_features.append(x)
        x = self.resnet.layer4(x)  # 32
        x = self.ppm(x)
        down_features.append(x)
        x = self.fpn_module(down_features)
        x = F.upsample(x, input_size, mode='bilinear')

        return x


class UperNet(nn.Module):
    def __init__(self, num_classes, layer,
                 fpn_dim=512, pretrained=False, pool_sizes=(1, 2, 3, 6)):
        super(UperNet, self).__init__()
        if layer not in [50, 101]:
            raise ValueError('Resnet-{} is not supported \n'
                             'Currently supported models are Resnet 50, 101'.format(layer))
        print('Model: UperNet_Resnet{}, Pretrained: {}'.format(
            layer, pretrained))
        if layer == 50:
            self.resnet = resnet50(pretrained=pretrained)
        elif layer == 101:
            self.resnet = resnet101(pretrained=pretrained)

        expansion = self.resnet.layer1[0].expansion  # 4
        in_dims = [64, 128, 256, 512]
        in_dims = [dim * expansion for dim in in_dims]
        ppm_indim = in_dims[-1]
        self.pool_sizes = pool_sizes
        self.ppm = PyramidPoolingModule(ppm_indim, self.pool_sizes)
        ppm_outdim = ppm_indim * 2
        in_dims[-1] = ppm_outdim
        self.fpn_module = FPNModule(num_classes, fpn_dim, in_dims=in_dims)

    def forward(self, x):
        input_size = x.size()[2:]
        down_features = []
        x = self.resnet.relu1(self.resnet.bn1(self.resnet.conv1(x)))
        x = self.resnet.relu2(self.resnet.bn2(self.resnet.conv2(x)))
        x = self.resnet.relu3(self.resnet.bn3(self.resnet.conv3(x)))
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)  # 4
        down_features.append(x)
        x = self.resnet.layer2(x)  # 8
        down_features.append(x)
        x = self.resnet.layer3(x)  # 16
        down_features.append(x)
        x = self.resnet.layer4(x)  # 32
        x = self.ppm(x)
        down_features.append(x)
        x = self.fpn_module(down_features)
        x = F.upsample(x, input_size, mode='bilinear')

        return x
