# Copyright 2018 Yiwen Shao

# Apache 2.0

import torch.nn as nn
import torch.nn.functional as F
import torchvision


class FCNResnet(nn.Module):
    def __init__(self, num_classes, scale=8, layer=18, pretrained=False):
        super(FCNResnet, self).__init__()
        if scale not in [8, 16, 32]:
            raise ValueError('Only scale factor 8, 16, 32 are supported\n'
                             'but given {}'.format(scale))
        if layer not in [18, 34, 50, 101, 152]:
            raise ValueError('Resnet-{} is not supported \n'
                             'Currently supported models are Resnet 18, 34, 50,\n'
                             '101, 152'.format(layer))
        print('Model: FCN{}s_Resnet{}, Pretrained: {}'.format(
            scale, layer, pretrained))
        self.scale = scale
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
        expansion = self.resnet.layer1[0].expansion
        self.score_32s = nn.Conv2d(512 * expansion,
                                   num_classes,
                                   kernel_size=1)
        if self.scale <= 16:
            self.score_16s = nn.Conv2d(256 * expansion,
                                       num_classes,
                                       kernel_size=1)
        if self.scale <= 8:
            self.score_8s = nn.Conv2d(128 * expansion,
                                      num_classes,
                                      kernel_size=1)

    def forward(self, x):
        input_size = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)

        x = self.resnet.layer2(x)
        if self.scale <= 8:
            score_8s = self.score_8s(x)

        x = self.resnet.layer3(x)
        if self.scale <= 16:
            score_16s = self.score_16s(x)

        x = self.resnet.layer4(x)
        score_32s = self.score_32s(x)

        if self.scale == 32:
            score = F.upsample_bilinear(score_32s, input_size)
        elif self.scale == 16:
            score_16s += F.upsample_bilinear(score_32s,
                                             score_16s.size()[2:])
            score = F.upsample_bilinear(score_16s, input_size)
        elif self.scale == 8:
            score_16s += F.upsample_bilinear(score_32s,
                                             score_16s.size()[2:])
            score_8s += F.upsample_bilinear(score_16s,
                                            score_8s.size()[2:])
            score = F.upsample_bilinear(score_8s, input_size)

        return score


class FCNVGG16(nn.Module):
    def __init__(self, num_classes, scale=8, pretrained=False):
        super(FCNVGG16, self).__init__()
        if scale not in [8, 16, 32]:
            raise ValueError('Only scale factor 8, 16, 32 are supported\n'
                             'but given {}'.format(scale))
        print('Model: FCN{}s_VGG16, Pretrained: {}'.format(scale, pretrained))
        self.scale = scale
        vgg16 = torchvision.models.vgg16(pretrained=pretrained)
        feature_layers = list(vgg16.features.children())
        self.block1 = nn.Sequential(*feature_layers[:5])
        self.block2 = nn.Sequential(*feature_layers[5:10])
        self.block3 = nn.Sequential(*feature_layers[10:17])
        self.block4 = nn.Sequential(*feature_layers[17:24])
        self.block5 = nn.Sequential(*feature_layers[24:31])

        self.score_32s = nn.Sequential(nn.Conv2d(512, 4096, 7),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout2d(),
                                       nn.Conv2d(4096, 4096, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout2d(),
                                       nn.Conv2d(4096, num_classes, 1))
        if self.scale <= 16:
            self.score_16s = nn.Conv2d(512, num_classes, 1)
        if self.scale <= 8:
            self.score_8s = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        input_size = x.size()[2:]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        if self.scale <= 8:
            score_8s = self.score_8s(x)

        x = self.block4(x)
        if self.scale <= 16:
            score_16s = self.score_16s(x)

        x = self.block5(x)
        score_32s = self.score_32s(x)

        if self.scale == 32:
            score = F.upsample_bilinear(score_32s, input_size)
        elif self.scale == 16:
            score_16s += F.upsample_bilinear(score_32s,
                                             score_16s.size()[2:])
            score = F.upsample_bilinear(score_16s, input_size)
        elif self.scale == 8:
            score_16s += F.upsample_bilinear(score_32s,
                                             score_16s.size()[2:])
            score_8s += F.upsample_bilinear(score_16s,
                                            score_8s.size()[2:])
            score = F.upsample_bilinear(score_8s, input_size)

        return score
