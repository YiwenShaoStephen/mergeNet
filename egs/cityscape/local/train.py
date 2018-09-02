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
from models.modules import DataParallelWithCallback
import torch.optim.lr_scheduler as lr_scheduler
from utils.dataset import AllDataset, ClassDataset, OffsetDataset
from utils.loss import CrossEntropyLossOneHot, SoftDiceLoss, MultiBCEWithLogitsLoss
from utils.train_utils import train, validate, sample, save_checkpoint, generate_offsets


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
                    help='log frequency for tensorboard (default: 1000)')
parser.add_argument('--visual-freq', default=0, type=int,
                    help='visualize network output every n epochs')
parser.add_argument('--gpu', default=-1, type=int, nargs='+',
                    help='gpu ids')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    help='mini-batch size (default: 16)')
parser.add_argument('--mode', default='all', type=str,
                    choices=['all', 'class', 'offset'],
                    help='training mode')
parser.add_argument('--crop-size', default=None, type=int,
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
parser.add_argument('--train-img', default='data/train', type=str,
                    help='Directory of training images')
parser.add_argument('--val-img', default='data/val', type=str,
                    help='Directory of validation images')
parser.add_argument('--train-ann', type=str,
                    default='data/annotations/instancesonly_filtered_gtFine_train.json',
                    help='Path to training set annotations')
parser.add_argument('--val-ann', type=str,
                    default='data/annotations/instancesonly_filtered_gtFine_val.json',
                    help='Path to validation set annotations')
parser.add_argument('--limits', default=None, type=int,
                    help="If given, is the size of subset we use for training")
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--pretrain', help='use pretrained model on ImageNet',
                    action='store_true')
parser.add_argument('--crop', help='Use cropped train images',
                    action='store_true')
parser.add_argument('--score', help='Use score metrics in training',
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

    # model configurations
    num_classes = args.num_classes
    num_offsets = args.num_offsets
    if args.mode == 'offset':  # offset only
        num_classes = 0
    if args.mode == 'class':  # class only
        num_offsets = 0

    # model
    model = get_model(num_classes, num_offsets, args.arch, args.pretrain)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_iou = checkpoint['best_iou']
            model.load_state_dict(checkpoint['model_state'])
            if 'offset' in checkpoint:  # class mode doesn't have offset
                offset_list = checkpoint['offset']
                print("offsets are: {}".format(offset_list))
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            raise ValueError(
                "=> no checkpoint found at '{}'".format(args.resume))

    # model distribution
    if args.gpu != -1:
        # DataParallel wrapper (synchronzied batchnorm edition)
        if len(args.gpu) > 1:
            model = DataParallelWithCallback(model, device_ids=args.gpu)
        model.cuda()

    # dataset
    if args.mode == 'all':
        offset_list = generate_offsets(80 / args.scale, args.num_offsets)
        trainset = AllDataset(args.train_img, args.train_ann, num_classes, offset_list,
                              scale=args.scale, crop=args.crop,
                              crop_size=(args.crop_size, args.crop_size),
                              limits=args.limits)
        valset = AllDataset(args.val_img, args.val_ann, num_classes, offset_list,
                            scale=args.scale, limits=args.limits)
        class_nms = trainset.catNms
    elif args.mode == 'class':
        offset_list = None
        trainset = ClassDataset(args.train_img, args.train_ann,
                                scale=args.scale, crop=args.crop,
                                crop_size=(args.crop_size, args.crop_size),
                                limits=args.limits)
        valset = ClassDataset(args.val_img, args.val_ann,
                              scale=args.scale, limits=args.limits)
        class_nms = trainset.catNms
    elif args.mode == 'offset':
        offset_list = generate_offsets(80 / args.scale, args.num_offsets)
        print("offsets are: {}".format(offset_list))
        trainset = OffsetDataset(args.train_img, args.train_ann, offset_list,
                                 scale=args.scale, crop=args.crop,
                                 crop_size=args.crop_size,
                                 limits=args.limits)
        valset = OffsetDataset(args.val_img, args.val_ann, offset_list,
                               scale=args.scale, limits=args.limits)
        class_nms = None

    trainloader = torch.utils.data.DataLoader(
        trainset, num_workers=4, batch_size=args.batch_size, shuffle=True)
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
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])

    # # define loss functions

    criterion_ofs = torch.nn.BCEWithLogitsLoss().cuda()

    if args.mode == 'all':
        criterion_cls = torch.nn.BCEWithLogitsLoss().cuda()
        criterion_ofs = torch.nn.BCEWithLogitsLoss().cuda()
    elif args.mode == 'class':
        criterion_cls = torch.nn.BCEWithLogitsLoss().cuda()
        criterion_ofs = None
    elif args.mode == 'offset':
        criterion_cls = None
        if args.loss == 'bce':
            print('Using Binary Cross Entropy Loss')
            criterion_ofs = torch.nn.BCEWithLogitsLoss().cuda()
        elif args.loss == 'mbce':
            print('Using Weighted Multiclass BCE Loss')
            criterion_ofs = MultiBCEWithLogitsLoss().cuda()
        elif args.loss == 'dice':
            print('Using Soft Dice Loss (0 mode)')
            criterion_ofs = SoftDiceLoss(mode='0').cuda()
        else:
            print('Using Cross Entropy Loss')
            criterion_ofs = CrossEntropyLossOneHot().cuda()

    # define learning rate scheduler
    if not args.milestones:
        milestones = [args.epochs]
    else:
        milestones = args.milestones
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.2, last_epoch=args.start_epoch - 1)

    # start iteration count
    iterations = args.start_epoch * int(len(trainset) / args.batch_size)

    # train
    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()
        iterations = train(trainloader, model, optimizer,
                           args.batch_size, epoch, iterations,
                           criterion_cls=criterion_cls, class_nms=class_nms,
                           criterion_ofs=criterion_ofs, offset_list=offset_list,
                           print_freq=args.print_freq,
                           log_freq=args.log_freq,
                           tensorboard=args.tensorboard,
                           score=args.score,
                           alpha=args.alpha)
        val_iou = validate(valloader, model,
                           args.batch_size, epoch, iterations,
                           criterion_cls=criterion_cls, class_nms=class_nms,
                           criterion_ofs=criterion_ofs, offset_list=offset_list,
                           print_freq=args.print_freq,
                           log_freq=args.log_freq,
                           tensorboard=args.tensorboard,
                           score=args.score,
                           alpha=args.alpha)
        # visualize some example outputs after each epoch
        if args.visual_freq > 0 and epoch % args.visual_freq == 0:
            outdir = '{}/imgs/{}'.format(args.dir, epoch)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            sample(model, valloader, outdir, num_classes,
                   num_offsets)

        # save checkpoint
        is_best = val_iou > best_iou
        best_iou = max(val_iou, best_iou)
        if args.gpu != -1 and len(args.gpu) > 1:
            state_dict = {'epoch': epoch + 1,
                          'model_state': model.module.state_dict(),  # remove 'module' in checkpoint
                          'best_iou': best_iou,
                          'optimizer': optimizer.state_dict()
                          }
        else:
            state_dict = {'epoch': epoch + 1,
                          'model_state': model.state_dict(),
                          'best_iou': best_iou,
                          'optimizer': optimizer.state_dict()
                          }
        if args.mode != 'class':
            state_dict['offset'] = offset_list
        save_checkpoint(args.dir, state_dict, is_best)

    print('Best validation mean iou: ', best_iou)


if __name__ == '__main__':
    main()
