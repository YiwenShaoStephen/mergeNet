# Copyright 2018 Yiwen Shao

# Apache 2.0

""" This script provides useful functions for doing inference.
"""

import os
import time
import torch
import numpy as np
import torch.nn.functional as F
from utils.train_utils import AverageMeter
from utils.score import runningScore, offsetIoU


def class_inference(dataloader, exp_dir, model, n_classes, batch_size, print_freq=10,
                    score=False, class_nms=None, class_map=None, tile_predict=False, gpu=False):
    """Perform class inference on the dataset"""
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()
    if score:
        score_metrics = runningScore(n_classes, class_nms)

    end = time.time()
    for i, vals in enumerate(dataloader):
        image_ids = vals[0]
        image_ids = image_ids.numpy()
        img = vals[1]
        if gpu:
            img = img.cuda()
        if score:  # we will need ground truth to score
            target = vals[2]
            class_mask = target[:, :n_classes, :, :]
        with torch.no_grad():
            if tile_predict:
                output = model.tile_predict(img, n_classes)
            else:
                output = model(img)
                output = F.sigmoid(output)

            # re-map class id to fit with the dataset loader
            if class_map:
                output_mapped = torch.zeros(output.shape)
                for j in range(len(class_map)):
                    new_id = class_map[j]
                    output_mapped[:, new_id, :, :] = output[:, j, :, :]
                output = output_mapped

            if score:
                score_metrics.update(output, class_mask)

            outdir = '{}/npy'.format(exp_dir)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            for k, image_id in enumerate(image_ids):
                save(output[k], outdir, str(image_id), suffix='class')
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % print_freq == 0:
                print('Val: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                          i, len(dataloader), batch_time=batch_time))

    if score:
        score, class_iou = score_metrics.get_scores()
        score_metrics.print_stat()


def offset_inference(dataloader, exp_dir, model, offset_list, batch_size,
                     print_freq=10, score=False, gpu=False):
    """Perform offset inference on the dataset"""
    batch_time = AverageMeter()
    n_offsets = len(offset_list)

    # switch to evaluate mode
    model.eval()

    if score:
        offset_metrics = offsetIoU(offset_list)

    end = time.time()
    for i, vals in enumerate(dataloader):
        image_ids = vals[0]
        image_ids = image_ids.numpy()
        img = vals[1]
        if gpu:
            img = img.cuda()
        if score:
            target = vals[2]
            bound_mask = target[:, -n_offsets:, :, :]
        with torch.no_grad():
            output = model(img)
            output = F.sigmoid(output)

            if score:
                offset_metrics.update(output, bound_mask)

            outdir = '{}/npy'.format(exp_dir)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            for k, image_id in enumerate(image_ids):
                save(output[k], outdir, str(image_id), suffix='offset')
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % print_freq == 0:
                print('Val: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                          i, len(dataloader), batch_time=batch_time))
                if score:
                    iou, mean_iou = offset_metrics.get_scores()
                    offset_metrics.print_stat()

    if score:
        iou, mean_iou = offset_metrics.get_scores()
        offset_metrics.print_stat()


def save(pred, outdir, name, suffix='class'):
    """save model output as np array for future use
    """
    filename = outdir + '/' + name + '.' + suffix + '.npy'
    np.save(filename, pred.cpu().numpy())
