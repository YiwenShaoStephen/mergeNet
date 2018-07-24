# Copyright 2018 Yiwen Shao

# Apache 2.0

import os
import argparse
import torch
from models import get_model
from utils.dataset import OffsetDataset
from utils.train_utils import generate_offsets
from utils.inference_utils import offset_inference

parser = argparse.ArgumentParser(
    description='Cityscapes offset inference')
parser.add_argument('--model', type=str, required=True,
                    help='model path.')
parser.add_argument('--dir', type=str, required=True,
                    help='output directory of inference result (numpy arrays)')
parser.add_argument('--img', type=str, default='data/val',
                    help='test images directory')
parser.add_argument('--ann', type=str,
                    default='data/annotations/instancesonly_filtered_gtFine_val.json',
                    help='path to test annotation or info file')
parser.add_argument('--mode', type=str, default='val')
parser.add_argument('--arch', type=str, default=None,
                    help='model architecture')
parser.add_argument('--gpu', help='use gpu',
                    action='store_true')
parser.add_argument('--score', help='do scoring when inference',
                    action='store_true')


def main():
    global args
    args = parser.parse_args()
    num_offsets = 10
    args.batch_size = 4
    model = get_model(0, num_offsets, args.arch)
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model,
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        #offset_list = checkpoint['offset']
        offset_list = generate_offsets(num_offsets)
        print("loaded.")
        print("offsets are: {}".format(offset_list))
    else:
        raise ValueError(
            "=> no checkpoint found at '{}'".format(args.model))

    valset = OffsetDataset(args.img, args.ann, offset_list,
                           mode='val', scale=2)
    dataloader = torch.utils.data.DataLoader(
        valset, num_workers=0, batch_size=args.batch_size)

    offset_inference(dataloader, args.dir, model, offset_list,
                     args.batch_size, score=args.score, gpu=args.gpu)


if __name__ == '__main__':
    main()
