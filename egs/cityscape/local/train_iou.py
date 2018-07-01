# Copyright 2018 Yiwen Shao

# Apache 2.0

""" This script trains the encoding network that the input images are of size
    c * h * w and the output feature maps are of size (num_class + num_offset) * h * w
"""

import sys
import torch
import argparse
import os
import random
from models import UNet, FCNResnet, FCNVGG16, PSPNet, PSPFPNet
import torch.optim.lr_scheduler as lr_scheduler
from utils.dataset import COCODataset
from utils.loss import CrossEntropyLossOneHot, SoftDiceLoss, MultiBCEWithLogitsLoss
from utils.core_config import CoreConfig
from utils.train_utils import train, validate, sample, save_checkpoint, soft_dice_loss, runningScore


parser = argparse.ArgumentParser(description='Pytorch cityscape setup')
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
                    help='log frequency is using tensorboard (default: 1000)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    help='mini-batch size (default: 16)')
parser.add_argument('--train-image-size', default=None, type=int,
                    help='The size of the parts of training images that we'
                    'train on (in order to form a fixed minibatch size).')
parser.add_argument('--loss', default='bce', type=str, choices=['bce', 'mbce', 'dice', 'ce'],
                    help='loss function')
parser.add_argument('--alpha', default=1, type=float,
                    help='weight of offset losses')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--milestones', default=None, nargs='+', type=int,
                    help='step decay position')
parser.add_argument('--arch', default='fcn16s', type=str,
                    help='model architecture')
parser.add_argument('--nesterov', default=True,
                    type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--train-img', default='data/train', type=str,
                    help='Directory of training images')
parser.add_argument('--val-img', default='data/val', type=str,
                    help='Directory of validation images')
parser.add_argument('--train-ann',
                    default='data/annotations/instancesonly_filtered_gtFine_train.json',
                    help='Path to training set annotations')
parser.add_argument('--val-ann',
                    default='data/annotations/instancesonly_filtered_gtFine_val.json',
                    help='Path to validation set annotations')
parser.add_argument('--limits', default=None, type=int,
                    help="If given, is the size of subset we use for training")
parser.add_argument('--class-name-file', default=None, type=str,
                    help="If given, is the subclass we are going to detect/segment")
parser.add_argument('--core-config', default='', type=str,
                    help='path of core configuration file')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--pretrain', help='use pretrained model on ImageNet',
                    action='store_true')
parser.add_argument('--visualize', help='Visualize network output after each epoch',
                    action='store_true')

best_iou = 0
random.seed(0)


def main():
    global args, best_iou, iterations
    args = parser.parse_args()

    if args.tensorboard:
        from tensorboard_logger import configure
        print("Using tensorboard")
        configure("%s" % (args.dir))

    # loading core configuration
    c_config = CoreConfig()
    if args.core_config == '':
        print('No core config file given, using default core configuration')
    if not os.path.exists(args.core_config):
        sys.exit('Cannot find the config file: {}'.format(args.core_config))
    else:
        c_config.read(args.core_config)
        print('Using core configuration from {}'.format(args.core_config))

    offset_list = c_config.offsets
    print("offsets are: {}".format(offset_list))

    # model configurations from core config
    num_classes = c_config.num_classes
    num_offsets = len(c_config.offsets)

    if args.class_name_file:
        with open(args.class_name_file, 'r') as fh:
            class_nms = fh.readline().split()
            print('Training on {} classes: {}'.format(
                len(class_nms), class_nms))
    else:
        class_nms = None
        print('Training on all classes.')

    # dataset
    trainset = COCODataset(args.train_img, args.train_ann, c_config,
                           (args.train_image_size, args.train_image_size * 2),
                           class_nms=class_nms, limits=args.limits)
    trainloader = torch.utils.data.DataLoader(
        trainset, num_workers=4, batch_size=args.batch_size, shuffle=True)
    valset = COCODataset(args.val_img, args.val_ann, c_config,
                         (args.train_image_size, args.train_image_size * 2),
                         class_nms=class_nms, limits=args.limits)
    valloader = torch.utils.data.DataLoader(
        valset, num_workers=4, batch_size=args.batch_size)
    NUM_TRAIN = len(trainset)
    NUM_VAL = len(valset)
    print('Training samples: {0} \n'
          'Validation samples: {1}'.format(NUM_TRAIN, NUM_VAL))

    # model
    valid_archs = ['fcn{}_resnet{}'.format(x, y)
                   for x in [8, 16, 32] for y in [18, 34, 50, 101, 152]]
    valid_archs += ['fcn{}_vgg16'.format(x) for x in [8, 16, 32]]
    valid_archs += ['unet']
    valid_archs += ['pspnet']
    valid_archs += ['pspfpnet']
    if args.arch not in valid_archs:
        raise ValueError('Supported models are: {} \n'
                         'but given {}'.format(valid_archs, args.arch))
    if args.arch == 'unet':
        model = UNet(num_classes, num_offsets)
    elif 'vgg16' in args.arch:
        names = args.arch.split('_')
        scale = int(names[0][3:])
        model = FCNVGG16(num_classes + num_offsets,
                         scale=scale, pretrained=args.pretrain)
    elif 'resnet' in args.arch:
        names = args.arch.split('_')
        scale = int(names[0][3:])
        layer = int(names[1][6:])
        model = FCNResnet(num_classes + num_offsets,
                          scale=scale, layer=layer, pretrained=args.pretrain)
    elif 'pspnet' in args.arch:
        layer = 101
        model = PSPNet(num_classes + num_offsets,
                       layer, pretrained=args.pretrain)
    elif 'fpnet' in args.arch:
        layer = 101
        model = PSPFPNet(num_classes + num_offsets, layer,
                         pretrained=args.pretrain)
    model = model.cuda()

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

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
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            raise ValueError(
                "=> no checkpoint found at '{}'".format(args.resume))

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
    score_metrics = runningScore(num_classes, valset.catNms)
    score_metrics_train = runningScore(num_classes, trainset.catNms)

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
                           alpha=args.alpha)
        val_iou = validate(valloader, model, criterion_cls, criterion_ofs,
                           num_classes, args.batch_size, epoch, iterations,
                           print_freq=args.print_freq,
                           log_freq=args.log_freq,
                           tensorboard=args.tensorboard,
                           score_metrics=score_metrics,
                           alpha=args.alpha)
        # visualize some example outputs after each epoch
        if args.visualize:
            outdir = '{}/imgs/{}'.format(args.dir, epoch + 1)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            sample(model, valloader, outdir, c_config)

        is_best = val_iou > best_iou
        best_iou = max(val_iou, best_iou)
        save_checkpoint(args.dir, {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_iou': best_iou,
            'optimizer': optimizer.state_dict(),
        }, is_best)
    print('Best validation mean iou: ', best_iou)


if __name__ == '__main__':
    main()
