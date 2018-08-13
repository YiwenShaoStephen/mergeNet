# Copyright 2018 Yiwen Shao

# Apache 2.0

""" This script provides useful functions for training a network in pytorch.
"""

import os
import shutil
import time
import math
import torch
import torchvision
from tensorboard_logger import log_value
from utils.score import runningScore, offsetIoU


def train(trainloader, model, optimizer, batch_size, epoch, iterations,
          criterion_cls=None, class_nms=None,
          criterion_ofs=None, offset_list=None,
          print_freq=10, log_freq=1000, tensorboard=True, score=False, alpha=1):
    """Train for one epoch on the training set"""
    model.train()
    if criterion_cls:
        n_classes = len(class_nms)
        cls_losses = AverageMeter()
        if score:
            score_metrics = runningScore(n_classes, class_nms)
    if criterion_ofs:
        ofs_losses = AverageMeter()
        if score:
            offset_metrics = offsetIoU(offset_list)
    all_losses = AverageMeter()
    batch_time = AverageMeter()

    # log learning rate to tensorboard
    if tensorboard:
        lr = optimizer.param_groups[0]['lr']
        log_value('learning_rate', lr, epoch)

    end = time.time()
    for i, (img, target) in enumerate(trainloader):
        img = img.cuda()
        target = target.cuda()
        if criterion_cls:
            class_target = target[:, :n_classes, :, :]
            if criterion_ofs:
                ofs_target = target[:, n_classes:, :, :]
        elif criterion_ofs:
            ofs_target = target

        prediction = model(img)  # forward network

        if criterion_cls:
            class_pred = prediction[:, :n_classes, :, :]
            if criterion_ofs:
                ofs_pred = prediction[:, n_classes:, :, :]
        elif criterion_ofs:
            ofs_pred = prediction

        optimizer.zero_grad()

        if criterion_cls:
            cls_loss = criterion_cls(class_pred, class_target)
            cls_losses.update(cls_loss.item(), batch_size)
        if criterion_ofs:
            ofs_loss = criterion_ofs(ofs_pred, ofs_target)
            ofs_losses.update(ofs_loss.item(), batch_size)

        if criterion_cls and criterion_ofs:  # both case
            all_loss = cls_loss + alpha * ofs_loss
        elif criterion_cls:  # class only case
            all_loss = cls_loss
        elif criterion_ofs:  # offset only case
            all_loss = ofs_loss

        all_losses.update(all_loss.item(), batch_size)
        all_loss.backward()
        optimizer.step()

        iterations += 1

        if criterion_cls and score:
            score_metrics.update(
                torch.sigmoid(class_pred), class_target)
        if criterion_ofs and score:
            offset_metrics.update(
                torch.sigmoid(ofs_pred), ofs_target)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                      epoch, i, len(trainloader), batch_time=batch_time))

        # log to TensorBoard
        if tensorboard and iterations % log_freq == 0:
            if criterion_cls:
                log_value('train_cls_loss', cls_losses.avg,
                          int(iterations / log_freq))
            if criterion_ofs:
                log_value('train_ofs_loss', ofs_losses.avg,
                          int(iterations / log_freq))

    if criterion_cls and score:
        scores, class_iou = score_metrics.get_scores()
        mean_iou = scores['mean_IU']
        if tensorboard:
            log_value('train_iou', mean_iou, epoch)
        score_metrics.print_stat()

    if criterion_ofs and score:
        iou, mean_iou = offset_metrics.get_scores()
        if tensorboard:
            log_value('train_ofs_miou', mean_iou, epoch)
            log_value('train_ofs_1_iou', iou[0], epoch)
            log_value('train_ofs_2_iou', iou[1], epoch)
        offset_metrics.print_stat()

    return iterations


def validate(validateloader, model, batch_size, epoch, iterations,
             criterion_cls=None, class_nms=None,
             criterion_ofs=None, offset_list=None,
             print_freq=10, log_freq=1000, tensorboard=True, score=False, alpha=1):
    """Perform validation on the validation set"""
    model.eval()
    if criterion_cls:
        n_classes = len(class_nms)
        cls_losses = AverageMeter()
        if score:
            score_metrics = runningScore(n_classes, class_nms)
    if criterion_ofs:
        ofs_losses = AverageMeter()
        if score:
            offset_metrics = offsetIoU(offset_list)
    all_losses = AverageMeter()
    batch_time = AverageMeter()

    end = time.time()
    for i, (img, target) in enumerate(validateloader):

        with torch.no_grad():
            img = img.cuda()
            target = target.cuda()
            if criterion_cls:
                class_target = target[:, :n_classes, :, :]
                if criterion_ofs:
                    ofs_target = target[:, n_classes:, :, :]
            elif criterion_ofs:
                ofs_target = target

            prediction = model(img)  # forward network

            if criterion_cls:
                class_pred = prediction[:, :n_classes, :, :]
                if criterion_ofs:
                    ofs_pred = prediction[:, n_classes:, :, :]
            elif criterion_ofs:
                ofs_pred = prediction

            if criterion_cls:
                cls_loss = criterion_cls(class_pred, class_target)
                cls_losses.update(cls_loss.item(), batch_size)
            if criterion_ofs:
                ofs_loss = criterion_ofs(ofs_pred, ofs_target)
                ofs_losses.update(ofs_loss.item(), batch_size)

            if criterion_cls and criterion_ofs:  # both case
                all_loss = cls_loss + alpha * ofs_loss
            elif criterion_cls:  # class only case
                all_loss = cls_loss
            elif criterion_ofs:  # offset only case
                all_loss = ofs_loss

            all_losses.update(all_loss.item(), batch_size)

            if criterion_cls and score:
                score_metrics.update(
                    torch.sigmoid(class_pred), class_target)
            if criterion_ofs and score:
                offset_metrics.update(
                    torch.sigmoid(ofs_pred), ofs_target)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % print_freq == 0:
                print('Val: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                          epoch, i, len(validateloader), batch_time=batch_time))

    # log to TensorBoard
    if tensorboard:
        if criterion_cls:
            log_value('val_cls_loss', cls_losses.avg,
                      int(iterations / log_freq))
        if criterion_ofs:
            log_value('val_ofs_loss', ofs_losses.avg,
                      int(iterations / log_freq))

    if criterion_cls and score:
        scores, class_iou = score_metrics.get_scores()
        mean_cls_iou = scores['mean_IU']
        if tensorboard:
            log_value('val_iou', mean_cls_iou, epoch)
        score_metrics.print_stat()

    if criterion_ofs and score:
        iou, mean_ofs_iou = offset_metrics.get_scores()
        if tensorboard:
            log_value('val_ofs_miou', mean_ofs_iou, epoch)
            log_value('val_ofs_1_iou', iou[0], epoch)
            log_value('val_ofs_2_iou', iou[1], epoch)
        offset_metrics.print_stat()

    if criterion_cls and criterion_ofs:
        mean_iou = mean_cls_iou + mean_ofs_iou
    elif criterion_cls:
        mean_iou = mean_cls_iou
    elif criterion_ofs:
        mean_iou = mean_ofs_iou

    return mean_iou


def sample(model, dataloader, outdir, n_classes, n_offsets):
    """Visualize some predicted masks and grond truth on data to get a better intuition
       about the performance.
    """
    if n_classes > 0:
        with_class = True
    else:
        with_class = False
    if n_offsets > 0:
        with_offset = True
    else:
        with_offset = False

    data_iter = iter(dataloader)
    img, target = data_iter.next()
    torchvision.utils.save_image(img, '{0}/raw.png'.format(outdir))
    if with_class:
        class_target = target[:, :n_classes, :, :]
        for i in range(n_classes):
            torchvision.utils.save_image(
                class_target[:, i:i + 1, :, :], '{0}/class_{1}.png'.format(outdir, i))
        if with_offset:
            ofs_target = target[:, n_classes:, :, :]
            for i in range(n_offsets):
                torchvision.utils.save_image(
                    ofs_target[:, i:i + 1, :, :], '{0}/bound_{1}.png'.format(outdir, i))

    elif with_offset:
        ofs_target = target
        for i in range(n_offsets):
            torchvision.utils.save_image(
                ofs_target[:, i:i + 1, :, :], '{0}/bound_{1}.png'.format(outdir, i))

    if next(model.parameters()).is_cuda:
        img = img.cuda()
    with torch.no_grad():
        predictions = torch.sigmoid(model(img))

    if with_class:
        class_pred = predictions[:, :n_classes, :, :]
        for i in range(n_classes):
            torchvision.utils.save_image(
                class_pred[:, i:i + 1, :, :], '{0}/class_{1}pred.png'.format(outdir, i))
        if with_offset:
            ofs_pred = predictions[:, n_classes:, :, :]
            for i in range(n_offsets):
                torchvision.utils.save_image(
                    ofs_pred[:, i:i + 1, :, :], '{0}/bound_{1}pred.png'.format(outdir, i))
    elif with_offset:
        ofs_pred = predictions
        for i in range(n_offsets):
            torchvision.utils.save_image(
                ofs_pred[:, i:i + 1, :, :], '{0}/bound_{1}pred.png'.format(outdir, i))


def save_checkpoint(dir, state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "%s/" % (dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '%s/' %
                        (dir) + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def generate_offsets(max_offset=20, num_offsets=10):
    offset_list = []
    angle = math.pi * 5 / 9  # 100 degrees: just over 90 degrees.
    triangle = max(abs(math.cos((num_offsets - 1) * angle)),
                   abs(math.sin((num_offsets - 1) * angle)))
    base = abs(max_offset / triangle)
    size_ratio = math.pow(base, 1 / float(num_offsets - 1))
    for n in range(num_offsets):
        x = int(round(math.cos(n * angle) * math.pow(size_ratio, n)))
        y = int(round(math.sin(n * angle) * math.pow(size_ratio, n)))
        offset_list.append((x, y))
    return offset_list
