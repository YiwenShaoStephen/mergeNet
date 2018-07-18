# Copyright 2018 Yiwen Shao

# Apache 2.0

""" This script provides scoring metrics.
"""
import numpy as np


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
            pflat = (1 - pred[:, i, :, :]
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
