# Copyright 2018 Yiwen Shao

# Apache 2.0

import os
import argparse
import torch
from models import pspnet

parser = argparse.ArgumentParser(
    description='Pspnet caffe model to pytorch model converter')
parser.add_argument('--caffe-model', type=str, required=True,
                    help='caffe model path')
parser.add_argument('--pytorch-model', type=str, required=True,
                    help='pytorch model path')
parser.add_argument('--dataset', type=str, default='cityscapes',
                    help='dataset name')
parser.add_argument('--gpu', help='store pytorch model with gpu',
                    action='store_true')


def main():
    global args
    args = parser.parse_args()
    model = pspnet(version=args.dataset)
    if os.path.isfile(args.caffe_model):
        model.load_pretrained_model(args.caffe_model)
        model.float()
        state = {'model_state': model.state_dict()}
        torch.save(state, args.pytorch_model)
        print('Finish coverting caffe model to pytorch model')
    else:
        raise ValueError(
            "=> no checkpoint found at '{}'".format(args.caffe_model))


if __name__ == '__main__':
    main()
