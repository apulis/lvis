import torch  # noqa
import torch.nn as nn
import torch.nn.functional as F  # noqa

from ..registry import LOSSES
from . import binary_cross_entropy
from .utils import weight_reduce_loss  # noqa


def multi_label_softmax_cross_entropy(pred,
                                      label,
                                      weight=None,
                                      reduction='mean',
                                      avg_factor=None):
    loss = -1 * F.log_softmax(pred, dim=1)
    # weight preprocessing
    weight[weight < 0.2] = 0
    weight[weight >= 0.2] = 1.0
    # weight normalization
    weight = weight / weight.sum(dim=1).view(-1, 1)
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss


@LOSSES.register_module
class MultiLabelLoss(nn.Module):

    def __init__(self, use_sigmoid=False, reduction='mean', loss_weight=1.0):
        super(MultiLabelLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        else:
            # multi-label softmax cross entropy
            self.cls_criterion = multi_label_softmax_cross_entropy

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls
