#!/usr/bin/env python3

# Copyright 2018 Yiwen Shao
# Apache 2.0

""" This script trains the encoding network that the input images are of size
    c * h * w and the output feature maps are of size (num_class + num_offset) * h * w
"""

import torch
import argparse
import os
import random
from models import get_model
import torch.optim.lr_scheduler as lr_scheduler
from utils.dataset import COCODataset
from utils.loss import CrossEntropyLossOneHot, SoftDiceLoss, MultiBCEWithLogitsLoss
from utils.train_utils import train, validate, sample, save_checkpoint, generate_offsets
from utils.score import runningScore, offsetIoU

parser = argparse.ArgumentParser(description='Pytorch COCO setup')
parser.add_argument('dir', type=str,
                    help='directory of output models and logs')
parser.add_argument('--epochs', default=10, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--log-freq', default=1000, type=int,
                    help='log frequency for tensorboard (default: 1000)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    help='mini-batch size (default: 16)')
parser.add_argument('--train-image-size', default=None, type=int,
                    help='crop image size in train')
parser.add_argument('--scale', default=1, type=int,
                    help='downsample image scale factor')
parser.add_argument('--loss', default='bce', type=str, choices=['bce', 'mbce', 'dice', 'ce'],
                    help='loss function')
parser.add_argument('--alpha', default=1, type=float,
                    help='weight of offset losses')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--milestones', default=None, nargs='+', type=int,
                    help='step decay position')
parser.add_argument('--arch', default='pspfpnet', type=str,
                    help='model architecture')
parser.add_argument('--num-classes', default=9, type=int,
                    help='number of classes')
parser.add_argument('--num-offsets', default=10, type=int,
                    help='number of offsets')
parser.add_argument('--nesterov', default=True,
                    type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--train-img', default='data/train2017', type=str,
                    help='Directory of training images')
parser.add_argument('--val-img', default='data/val2017', type=str,
                    help='Directory of validation images')
parser.add_argument('--train-ann', type=str,
                    default='data/annotations/instances_train2017.json',
                    help='Path to training set annotations')
parser.add_argument('--val-ann', type=str,
                    default='data/annotations/instances_val2017.json',
                    help='Path to validation set annotations')
parser.add_argument('--limits', default=None, type=int,
                    help="If given, is the size of subset we use for training")
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--pretrain', help='use pretrained model on ImageNet',
                    action='store_true')
parser.add_argument('--visualize', help='Visualize network output after each epoch',
                    action='store_true')
parser.add_argument('--crop', help='Use cropped train images',
                    action='store_true')


best_loss = 1
random.seed(0)


def main():
    global args, best_iou, iterations
    args = parser.parse_args()

    if args.tensorboard:
        from tensorboard_logger import configure
        print("Using tensorboard")
        configure("%s" % (args.dir))

    offset_list = generate_offsets(args.num_offsets)

    # model configurations
    num_classes = args.num_classes
    num_offsets = args.num_offsets

    # model
    model = get_model(num_classes, num_offsets, args.arch, args.pretrain)
    model = model.cuda()

    # dataset
    trainset = COCODataset(args.train_img, args.train_ann, num_classes, offset_list,
                           scale=args.scale,
                           size=(args.train_image_size, args.train_image_size),
                           limits=args.limits, crop=args.crop)
    trainloader = torch.utils.data.DataLoader(
        trainset, num_workers=4, batch_size=args.batch_size, shuffle=True)
    valset = COCODataset(args.val_img, args.val_ann, num_classes, offset_list,
                         scale=args.scale, limits=args.limits)
    valloader = torch.utils.data.DataLoader(
        valset, num_workers=4, batch_size=4)
    num_train = len(trainset)
    num_val = len(valset)
    print('Training samples: {0} \n'
          'Validation samples: {1}'.format(num_train, num_val))

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, nesterov=args.nesterov,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_iou = checkpoint['best_iou']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            offset_list = checkpoint['offset']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            raise ValueError(
                "=> no checkpoint found at '{}'".format(args.resume))
    print("offsets are: {}".format(offset_list))

    # define loss functions
    if args.loss == 'bce':
        print('Using Binary Cross Entropy Loss')
        criterion_cls = torch.nn.BCEWithLogitsLoss().cuda()
    elif args.loss == 'mbce':
        print('Using Weighted Multiclass BCE Loss')
        criterion_cls = MultiBCEWithLogitsLoss().cuda()
    elif args.loss == 'dice':
        print('Using Soft Dice Loss')
        criterion_cls = SoftDiceLoss().cuda()
    else:
        print('Using Cross Entropy Loss')
        criterion_cls = CrossEntropyLossOneHot().cuda()

    criterion_ofs = torch.nn.BCEWithLogitsLoss().cuda()

    # define learning rate scheduler
    if not args.milestones:
        milestones = [args.epochs]
    else:
        milestones = args.milestones
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.1, last_epoch=args.start_epoch - 1)

    # start iteration count
    iterations = args.start_epoch * int(len(trainset) / args.batch_size)

    # define score metrics
    score_metrics_train = runningScore(num_classes, trainset.catNms)
    score_metrics = runningScore(num_classes, valset.catNms)
    offset_metrics_train = offsetIoU(offset_list)
    offset_metrics_val = offsetIoU(offset_list)

    # train
    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()
        iterations = train(trainloader, model, criterion_cls, criterion_ofs,
                           optimizer, num_classes,
                           args.batch_size, epoch, iterations,
                           print_freq=args.print_freq,
                           log_freq=args.log_freq,
                           tensorboard=args.tensorboard,
                           score_metrics=score_metrics_train,
                           offset_metrics=offset_metrics_train,
                           alpha=args.alpha)
        val_iou = validate(valloader, model, criterion_cls, criterion_ofs,
                           num_classes, args.batch_size, epoch, iterations,
                           print_freq=args.print_freq,
                           log_freq=args.log_freq,
                           tensorboard=args.tensorboard,
                           score_metrics=score_metrics,
                           offset_metrics=offset_metrics_val,
                           alpha=args.alpha)
        # visualize some example outputs after each epoch
        if args.visualize:
            outdir = '{}/imgs/{}'.format(args.dir, epoch + 1)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            sample(num_classes, num_offsets, model, valloader, outdir)

        is_best = val_iou > best_iou
        best_iou = max(val_iou, best_iou)
        save_checkpoint(args.dir, {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_iou': best_iou,
            'optimizer': optimizer.state_dict(),
            'offset': offset_list,
        }, is_best)
    print('Best validation mean iou: ', best_iou)


if __name__ == '__main__':
    main()
