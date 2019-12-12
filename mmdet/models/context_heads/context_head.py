import torch
import torch.nn as nn

from mmdet.core import force_fp32
from ..builder import build_loss
from ..plugins import ChannelGate
from ..registry import HEADS


@HEADS.register_module
class ContextHead(nn.Module):

    def __init__(self, in_channels, num_classes, cfg_channel_gate,
                 supervise_neg_categories, loss):
        super(ContextHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.supervise_neg_categories = supervise_neg_categories

        self.channel_gate = ChannelGate(self.in_channels, **cfg_channel_gate)
        self.fc_cls = nn.Linear(in_channels, num_classes)
        self.context_loss = build_loss(loss)

    def init_weights(self):
        for m in self.fc_cls.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        pool_feats, scale = self.channel_gate(x)
        return scale, self.fc_cls(pool_feats)

    def get_target(self, gt_pos_labels, img_metas):
        num_imgs = len(gt_pos_labels)
        if self.supervise_neg_categories:
            gt_neg_labels = self.get_neg_cat_ids(img_metas)  # noqa
            raise NotImplementedError
        else:
            labels = gt_pos_labels[0].new_zeros((num_imgs, self.num_classes),
                                                dtype=torch.long)
            for i, gt_pos_label in enumerate(gt_pos_labels):
                labels[i][list(set((gt_pos_label - 1).tolist()))] = 1
            label_weights = labels.new_ones(self.num_classes, dtype=torch.long)
        return labels, label_weights

    @force_fp32(apply_to=('cls_score'))
    def loss(self, cls_score, labels, label_weights, reduction_override=None):
        losses = dict()
        loss = self.context_loss(
            cls_score, labels, label_weights, avg_factor=None)
        losses['context_loss'] = loss
        return losses

    def get_neg_cat_ids(self, img_metas):
        neg_cat_ids = []
        for i, img_meta in enumerate(img_metas):
            neg_cat_ids.append(img_meta['neg_category_ids'])
        return neg_cat_ids

    # TODO
    # supervise loss function.
    # construct supervise label
    # loss function in code level
    # test and evaluation pipeline
    # more complex model
