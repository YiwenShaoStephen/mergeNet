# Copyright 2018 Yiwen Shao

# Apache 2.0

""" This script provides useful functions for training a network in pytorch.
"""

import os
import shutil
import time
import math
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from tensorboard_logger import log_value


def train(trainloader, model, criterion_cls, criterion_ofs, optimizer,
          n_classes, batch_size, epoch, iterations,
          print_freq=10, log_freq=1000, tensorboard=True, score_metrics=None,
          offset_metrics=None, alpha=1):
    """Train for one epoch on the training set"""
    model.train()
    cls_losses = AverageMeter()
    ofs_losses = AverageMeter()
    all_losses = AverageMeter()
    batch_time = AverageMeter()

    # log learning rate to tensorboard
    if tensorboard:
        lr = optimizer.param_groups[0]['lr']
        log_value('learning_rate', lr, epoch)

    if score_metrics:
        score_metrics.reset()  # clean up data
    if offset_metrics:
        offset_metrics.reset()

    end = time.time()
    for i, (input, target) in enumerate(trainloader):
        input = input.cuda()
        target = target.cuda()
        class_mask = target[:, :n_classes, :, :]
        bound_mask = target[:, n_classes:, :, :]
        output = model(input)

        optimizer.zero_grad()

        cls_loss = criterion_cls(output[:, :n_classes, :, :], class_mask)
        ofs_loss = criterion_ofs(output[:, n_classes:, :, :], bound_mask)
        all_loss = cls_loss + alpha * ofs_loss

        cls_losses.update(cls_loss.item(), batch_size)
        ofs_losses.update(ofs_loss.item(), batch_size)
        all_losses.update(all_loss.item(), batch_size)

        all_loss.backward()
        optimizer.step()

        iterations += 1

        if score_metrics:
            score_metrics.update(output[:, :n_classes, :, :], class_mask)
        if offset_metrics:
            offset_metrics.update(output[:, n_classes:, :, :], bound_mask)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Class Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                  'Offset Loss {ofs_loss.val:.4f} ({ofs_loss.avg:.4f})\t'.format(
                      epoch, i, len(trainloader), batch_time=batch_time,
                      cls_loss=cls_losses, ofs_loss=ofs_losses))

        # log to TensorBoard
        if tensorboard and iterations % log_freq == 0:
            log_value('train_cls_loss', cls_losses.avg,
                      int(iterations / log_freq))
            log_value('train_ofs_loss', ofs_losses.avg,
                      int(iterations / log_freq))

    if score_metrics:
        score, class_iou = score_metrics.get_scores()
        mean_iou = score['mean_IU']
        if tensorboard:
            log_value('train_iou', mean_iou, epoch)
        score_metrics.print_stat()

    if offset_metrics:
        iou, mean_iou = offset_metrics.get_scores()
        if tensorboard:
            log_value('train_ofs_miou', mean_iou, epoch)
            log_value('train_ofs_1_iou', iou[0], epoch)
            log_value('train_ofs_2_iou', iou[1], epoch)
        offset_metrics.print_stat()

    return iterations


def validate(validateloader, model, criterion_cls, criterion_ofs,
             n_classes, batch_size, epoch, iterations,
             print_freq=10, log_freq=1000, tensorboard=True, score_metrics=None,
             offset_metrics=None, alpha=1):
    """Perform validation on the validation set"""
    cls_losses = AverageMeter()
    ofs_losses = AverageMeter()
    all_losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    if score_metrics:
        score_metrics.reset()
    if offset_metrics:
        offset_metrics.reset()

    for i, (input, target) in enumerate(validateloader):

        with torch.no_grad():
            input = input.cuda()
            target = target.cuda()
            class_mask = target[:, :n_classes, :, :]
            bound_mask = target[:, n_classes:, :, :]

            output = model(input)

            cls_loss = criterion_cls(output[:, :n_classes, :, :], class_mask)
            ofs_loss = criterion_ofs(output[:, n_classes:, :, :], bound_mask)
            all_loss = cls_loss + alpha * ofs_loss

            cls_losses.update(cls_loss.item(), batch_size)
            ofs_losses.update(ofs_loss.item(), batch_size)
            all_losses.update(all_loss.item(), batch_size)

            if score_metrics:
                score_metrics.update(output[:, :n_classes, :, :], class_mask)
            if offset_metrics:
                offset_metrics.update(output[:, n_classes:, :, :], bound_mask)

            if i % print_freq == 0:
                print('Val: [{0}][{1}/{2}]\t'
                      'Class Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                      'Offset Loss {ofs_loss.val:.4f} ({ofs_loss.avg:.4f})\t'.format(
                          epoch, i, len(validateloader),
                          cls_loss=cls_losses, ofs_loss=ofs_losses))

    # log to TensorBoard
    if tensorboard:
        log_value('val_cls_loss', cls_losses.avg, int(iterations / log_freq))
        log_value('val_ofs_loss', ofs_losses.avg, int(iterations / log_freq))

    if offset_metrics:
        iou, mean_iou = offset_metrics.get_scores()
        if tensorboard:
            log_value('val_ofs_miou', mean_iou, epoch)
            log_value('val_ofs_1_iou', iou[0], epoch)
            log_value('val_ofs_2_iou', iou[1], epoch)
        offset_metrics.print_stat()

    if score_metrics:
        score, class_iou = score_metrics.get_scores()
        mean_iou = score['mean_IU']
        if tensorboard:
            log_value('val_iou', mean_iou, epoch)
        score_metrics.print_stat()

        return mean_iou

    return all_losses.avg


def sample(model, dataloader, outdir, core_config):
    """Visualize some predicted masks on training data to get a better intuition
       about the performance.
    """
    data_iter = iter(dataloader)
    img, classification, bound = data_iter.next()
    torchvision.utils.save_image(img, '{0}/raw.png'.format(outdir))
    for i in range(len(core_config.offsets)):
        torchvision.utils.save_image(
            bound[:, i:i + 1, :, :], '{0}/bound_{1}.png'.format(outdir, i))
    for i in range(core_config.num_classes):
        torchvision.utils.save_image(
            classification[:, i:i + 1, :, :], '{0}/class_{1}.png'.format(outdir, i))
    if next(model.parameters()).is_cuda:
        img = img.cuda()
    with torch.no_grad():
        predictions = model(img)
    predictions = predictions.detach()
    class_pred = predictions[:, :core_config.num_classes, :, :]
    bound_pred = predictions[:, core_config.num_classes:, :, :]
    for i in range(len(core_config.offsets)):
        torchvision.utils.save_image(
            bound_pred[:, i:i + 1, :, :], '{0}/bound_{1}pred.png'.format(outdir, i))
    for i in range(core_config.num_classes):
        torchvision.utils.save_image(
            class_pred[:, i:i + 1, :, :], '{0}/class_{1}pred.png'.format(outdir, i))

    return img, class_pred, bound_pred


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


class soft_dice_loss(torch.nn.Module):
    def __init__(self, smooth=1):
        super(soft_dice_loss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        iflat = inputs.view(-1)
        tflat = targets.view(-1)
        intersection = (iflat * tflat)
        loss = 1 - (2. * intersection.sum() + self.smooth) / \
            (iflat.sum() + tflat.sum() + self.smooth)
        return loss


class runningScore(object):
    """ Adapted from score written by wkentaro
        https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
    """

    def __init__(self, n_classes, class_nms):
        self.n_classes = n_classes
        self.class_nms = class_nms
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
        return hist

    def update(self, label_preds, label_truths):
        gt = label_truths[:, :self.n_classes, :, :].max(1)[1].cpu().numpy()
        pred = label_preds[:, :self.n_classes, :, :].max(1)[1].cpu().numpy()
        for lt, lp in zip(gt, pred):
            self.confusion_matrix += self._fast_hist(
                lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) +
                              hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(self.class_nms, iu))

        return {'overall_acc': acc,
                'mean_acc': acc_cls,
                'freq_acc': fwavacc,
                'mean_IU': mean_iu, }, cls_iu

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

    def print_stat(self):
        score, class_iou = self.get_scores()
        print('class\t IoU')
        for class_nm in self.class_nms:
            print('{}\t{}'.format(class_nm, class_iou[class_nm]))
        print('mean IoU\t{}'.format(score['mean_IU']))
        print('pixel acc\t{}'.format(score['overall_acc']))


class offsetIoU(object):
    def __init__(self, offset_list):
        self.offset_list = offset_list
        self.num_offsets = len(offset_list)
        self.intersection = np.zeros(self.num_offsets)
        self.union = np.zeros(self.num_offsets)
        self.iou = np.zeros(self.num_offsets)

    def update(self, pred, gt):
        # 1-0 convert
        for i in range(self.num_offsets):
            pflat = (1 - F.sigmoid(pred[:, i, :, :])
                     ).detach().cpu().contiguous().view(-1).numpy()
            gflat = (1 - gt[:, i, :, :]
                     ).detach().cpu().contiguous().view(-1).numpy()
            intersection = (pflat * gflat).sum()
            self.intersection[i] += intersection
            self.union[i] += pflat.sum() + gflat.sum() - intersection

    def reset(self):
        self.intersection = np.zeros(self.num_offsets)
        self.union = np.zeros(self.num_offsets)
        self.iou = np.zeros(self.num_offsets)

    def get_scores(self):
        for i in range(self.num_offsets):
            self.iou[i] = self.intersection[i] / self.union[i]
        return self.iou, self.iou.mean()

    def print_stat(self):
        iou, miou = self.get_scores()
        print('offset\t IoU')
        for i, offset in enumerate(self.offset_list):
            print('{}\t{}'.format(offset, iou[i]))
        print('mean IoU\t {}'.format(miou))


def generate_offsets(num_offsets=15):
    offset_list = []
    size_ratio = 1.4
    angle = math.pi * 5 / 9  # 100 degrees: just over 90 degrees.
    for n in range(num_offsets):
        x = round(math.cos(n * angle) * math.pow(size_ratio, n))
        y = round(math.sin(n * angle) * math.pow(size_ratio, n))
        offset_list.append((x, y))
    return offset_list
