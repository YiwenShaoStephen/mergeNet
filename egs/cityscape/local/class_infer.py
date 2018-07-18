# Copyright 2018 Yiwen Shao

# Apache 2.0

import os
import argparse
import torch
from models import get_model, pspnet
from utils.dataset import ClassDataset
from utils.inference_utils import class_inference

parser = argparse.ArgumentParser(
    description='Cityscapes class inference')
parser.add_argument('--model', type=str, required=True,
                    help='model path.')
parser.add_argument('--dir', type=str, required=True,
                    help='output directory of inference result (numpy arrays)')
parser.add_argument('--img', type=str, default='data/val',
                    help='test images directory')
parser.add_argument('--ann', type=str,
                    default='data/annotations/instancesonly_filtered_gtFine_val.json',
                    help='path to test annotation or info file')
parser.add_argument('--arch', type=str, default=None,
                    help='model architecture')
parser.add_argument('--gpu', type=int, default=-1, nargs='+',
                    help='gpu ids',
                    action='store_true')
parser.add_argument('--score', help='do scoring when inference',
                    action='store_true')
parser.add_argument('--caffe', help='use the converted pytorch model from caffe',
                    action='store_true')


def main():
    global args
    args = parser.parse_args()
    num_classes = 9
    args.batch_size = 4
    if args.caffe:
        model = pspnet(version='cityscapes')
        if os.path.isfile(args.model):
            print("=> loading checkpoint '{}'".format(args.model))
            checkpoint = torch.load(args.model)
            model.load_state_dict(checkpoint['model_state'])
            print("loaded.")
        else:
            raise ValueError(
                "=> no checkpoint found at '{}'".format(args.model))
    else:
        model = get_model(num_classes, 0, args.arch)
        if os.path.isfile(args.model):
            print("=> loading checkpoint '{}'".format(args.model))
            checkpoint = torch.load(args.model,
                                    map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            print("loaded.")
        else:
            raise ValueError(
                "=> no checkpoint found at '{}'".format(args.model))

    valset = ClassDataset(args.img, args.ann,
                          mode='val', caffe=args.caffe)
    dataloader = torch.utils.data.DataLoader(
        valset, num_workers=0, batch_size=args.batch_size)

    if args.caffe:  # re-map class id and do tile prediction
        class_map = [0, 1, 7, 6, 4, 3, 8, 2, 5]
        tile_predict = True
    else:
        class_map = None
        tile_predict = False

    if args.gpu != -1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu)
        use_gpu = True
    else:
        use_gpu = False

    class_inference(dataloader, args.dir, model, num_classes, args.batch_size,
                    score=args.score, class_nms=valset.catNms, class_map=class_map,
                    tile_predict=tile_predict, gpu=use_gpu)


if __name__ == '__main__':
    main()
