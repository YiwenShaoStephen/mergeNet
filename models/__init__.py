from __future__ import absolute_import

from .Unet import *
from .fcn import *
from .pspnet import *
from .pspnet_caffe import pspnet


def get_model(num_classes, num_offsets, arch, pretrain=False):
    valid_archs = ['fcn{}_resnet{}'.format(x, y)
                   for x in [8, 16, 32] for y in [18, 34, 50, 101, 152]]
    valid_archs += ['fcn{}_vgg16'.format(x) for x in [8, 16, 32]]
    valid_archs += ['unet']
    valid_archs += ['pspnet']
    valid_archs += ['pspfpnet']
    if arch not in valid_archs:
        raise ValueError('Supported models are: {} \n'
                         'but given {}'.format(valid_archs, arch))
    if arch == 'unet':
        model = UNet(num_classes, num_offsets)
    elif 'vgg16' in arch:
        names = arch.split('_')
        scale = int(names[0][3:])
        model = FCNVGG16(num_classes + num_offsets,
                         scale=scale, pretrained=pretrain)
    elif 'resnet' in arch:
        names = arch.split('_')
        scale = int(names[0][3:])
        layer = int(names[1][6:])
        model = FCNResnet(num_classes + num_offsets,
                          scale=scale, layer=layer, pretrained=pretrain)
    elif 'fpnet' in arch:
        layer = 101
        model = PSPFPNet(num_classes + num_offsets, layer,
                         pretrained=pretrain)
    elif 'pspnet' in arch:
        layer = 101
        model = PSPNet(num_classes + num_offsets,
                       layer, pretrained=pretrain)
    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    return model
