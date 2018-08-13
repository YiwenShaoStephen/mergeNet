import torch
import torch.nn.functional as F


def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"


class _Loss(torch.nn.Module):
    def __init__(self, size_average=True, reduce=True):
        super(_Loss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce


class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=True, reduce=True):
        super(_WeightedLoss, self).__init__(size_average, reduce)
        self.register_buffer('weight', weight)


class CrossEntropyLossOneHot(_WeightedLoss):
    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True):
        super(CrossEntropyLossOneHot, self).__init__(weight, size_average)
        self.ignore_index = ignore_index
        self.reduce = reduce

    def forward(self, input, target):
        _assert_no_grad(target)
        # n,c,h,w --> n,h,w; onehot to class label
        target_label = target.max(1)[1]
        return F.cross_entropy(input, target_label, self.weight, self.size_average,
                               self.ignore_index, self.reduce)


class SoftDiceLoss(torch.nn.Module):
    def __init__(self, mode='1', smooth=1):
        super(SoftDiceLoss, self).__init__()
        self.mode = mode
        self.smooth = smooth

    def forward(self, input, target):
        assert(input.shape == target.shape)
        loss = 0
        n, c, h, w = input.shape
        input_prob = torch.sigmoid(input)
        for i in range(c):
            iflat = input_prob[:, i, :, :].contiguous().view(-1)
            tflat = target[:, i, :, :].contiguous().view(-1)
            if self.mode == '0':  # '0' is of more importance
                iflat = 1 - iflat
                tflat = 1 - tflat
            intersection = (iflat * tflat)
            loss += 1 - (2. * intersection.sum() + self.smooth) / \
                (iflat.sum() + tflat.sum() + self.smooth)
        return loss


# class MultiBCEWithLogitsLoss()

class MultiBCEWithLogitsLoss(_Loss):
    def __init__(self, size_average=True, reduce=True):
        super(MultiBCEWithLogitsLoss, self).__init__(size_average, reduce)

    def forward(self, input, target):
        b, c, h, w = target.size()
        n = h * w
        weight = (n - F.sigmoid(input).sum(dim=-1).sum(dim=-1) + 1) / \
            (F.sigmoid(input).sum(dim=-1).sum(dim=-1) + 1)
        weight.unsqueeze_(-1).unsqueeze_(-1)
        weight = weight.expand(target.size())
        weight = weight * target + (1 - target)
        return F.binary_cross_entropy_with_logits(input, target,
                                                  weight, self.size_average, self.reduce)


class WeightedBCEWithLogitsLoss(_Loss):
    def __init__(self, size_average=True, reduce=True, alpha=0.5):
        super(WeightedBCEWithLogitsLoss, self).__init__(size_average, reduce)
        self.alpha = alpha

    def forward(self, input, target):
        weight = self.alpha * target + (1 - self.alpha) * (1 - target)
        return F.binary_cross_entropy_with_logits(input, target,
                                                  weight, self.size_average, self.reduce)
