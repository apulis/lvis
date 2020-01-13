import torch

from ..utils import multi_apply
from .transforms import bbox2delta


def bbox_target(pos_bboxes_list,
                neg_bboxes_list,
                pos_gt_bboxes_list,
                pos_gt_labels_list,
                cfg,
                reg_classes=1,
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0],
                concat=True):
    # pos_bboxes_list: each element corresponds to each batch
    labels, label_weights, bbox_targets, bbox_weights = multi_apply(
        bbox_target_single,
        pos_bboxes_list,
        neg_bboxes_list,
        pos_gt_bboxes_list,
        pos_gt_labels_list,
        cfg=cfg,
        reg_classes=reg_classes,
        target_means=target_means,
        target_stds=target_stds)

    if concat:
        labels = torch.cat(labels, 0)
        label_weights = torch.cat(label_weights, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        bbox_weights = torch.cat(bbox_weights, 0)
    return labels, label_weights, bbox_targets, bbox_weights


def bbox_target_single(pos_bboxes,
                       neg_bboxes,
                       pos_gt_bboxes,
                       pos_gt_labels,
                       cfg,
                       reg_classes=1,
                       target_means=[.0, .0, .0, .0],
                       target_stds=[1.0, 1.0, 1.0, 1.0]):
    num_pos = pos_bboxes.size(0)
    num_neg = neg_bboxes.size(0)
    num_samples = num_pos + num_neg
    labels = pos_bboxes.new_zeros(num_samples, dtype=torch.long)
    label_weights = pos_bboxes.new_zeros(num_samples)
    bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
    bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
    if num_pos > 0:
        labels[:num_pos] = pos_gt_labels
        pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
        label_weights[:num_pos] = pos_weight
        pos_bbox_targets = bbox2delta(pos_bboxes, pos_gt_bboxes, target_means,
                                      target_stds)
        bbox_targets[:num_pos, :] = pos_bbox_targets
        bbox_weights[:num_pos, :] = 1
    if num_neg > 0:
        label_weights[-num_neg:] = 1.0
    # one batch -> length of 512
    return labels, label_weights, bbox_targets, bbox_weights


def expand_target(bbox_targets, bbox_weights, labels, num_classes):
    bbox_targets_expand = bbox_targets.new_zeros(
        (bbox_targets.size(0), 4 * num_classes))
    bbox_weights_expand = bbox_weights.new_zeros(
        (bbox_weights.size(0), 4 * num_classes))
    for i in torch.nonzero(labels > 0).squeeze(-1):
        start, end = labels[i] * 4, (labels[i] + 1) * 4
        bbox_targets_expand[i, start:end] = bbox_targets[i, :]
        bbox_weights_expand[i, start:end] = bbox_weights[i, :]
    return bbox_targets_expand, bbox_weights_expand


def _expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 0).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds]] = 1
    if label_weights is None:
        bin_label_weights = None
    else:
        bin_label_weights = label_weights.view(-1, 1).expand(
            label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


def equalize_single(labels,
                    label_weights,
                    img_meta,
                    eql_cfg=None,
                    num_classes=1230 + 1):
    assert eql_cfg is not None
    supervise_cat_ids = eql_cfg['supervise_cat_ids']
    gt_neg_labels = img_meta['neg_category_ids']
    bin_labels, _ = _expand_binary_labels(labels, None, num_classes)
    bin_label_weights = labels.new_zeros(bin_labels.size(), dtype=torch.long)

    # process for negative proposals
    bin_label_weights[labels == 0, :] = 1

    # process for positibve proposals
    pos_inds = torch.nonzero(labels).squeeze(0).tolist()
    bin_label_weights[:, supervise_cat_ids] = 1
    bin_label_weights[:, gt_neg_labels] = 1
    for pos_ind in pos_inds:
        bin_label_weights[pos_ind, [0, labels[pos_ind].item()]] = 1
    return bin_labels, bin_label_weights


def process_class_label(labels,
                        label_weights,
                        img_meta_list,
                        concat_targets=False,
                        sparse_label=False,
                        graph=None,
                        use_sigmoid_cls=False,
                        eql_cfg=None,
                        num_classes=1230 + 1):
    if not concat_targets:
        assert isinstance(labels, list)
        if eql_cfg is not None:
            # equalize
            bin_labels, bin_label_weights = multi_apply(
                equalize_single,
                labels,
                label_weights,
                img_meta_list,
                eql_cfg=eql_cfg,
                num_classes=num_classes)
            bin_labels = torch.cat(bin_labels)
            bin_label_weights = torch.cat(bin_label_weights)
            target_meta = {'labels': torch.cat(labels)}
            return bin_labels, bin_label_weights, target_meta
    target_meta = {'labels': labels}
    if not sparse_label:
        return labels, label_weights, None
    # sparse labels: sigmoid
    bin_labels, bin_label_weights = _expand_binary_labels(
        labels, label_weights, num_classes)
    if graph is None:
        return bin_labels, bin_label_weights, target_meta
    # propagate on graph.
    pos_inds = labels > 0
    neg_inds = labels == 0
    label_weights = bin_labels.new_zeros(bin_labels.size(), dtype=torch.float)

    label_weights[pos_inds, 1:] = torch.matmul(
        bin_labels[pos_inds, 1:].float(), graph)
    label_weights[neg_inds, 0] = 1.0
    labels = (label_weights.clone() > 0).long()
    if use_sigmoid_cls:
        label_weights[label_weights == 0] = 1.0
        # label_weights = label_weights.new_ones(
        #     label_weights.size(), dtype=torch.float)
    return labels, label_weights, target_meta
