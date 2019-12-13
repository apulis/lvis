from __future__ import division
from collections import OrderedDict

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel

from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for name in log_vars:
        log_vars[name] = log_vars[name].item()

    return loss, log_vars


def main():
    cfg_file = 'configs/lvis/exp/mask-rcnn-FPN-50_GCM-avg_GCE_PRS-1e-2.py'
    cfg = Config.fromfile(cfg_file)
    cfg.gpus = 1

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    model = MMDataParallel(model, device_ids=range(1)).cuda()
    dataset = [build_dataset(cfg.data.train)]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            cfg.gpus,
            dist=False) for ds in dataset
    ]
    model.CLASSES = dataset[0].CLASSES

    for i, data in enumerate(data_loaders[0]):
        outputs = model(**data)
        loss, log_vars = parse_losses(outputs)
        print('iter: {}   '.format(i+1))
        for key, val in log_vars.items():
            print('{}: {:.3f}  '.format(key, val), end='')
        print('')


if __name__ == '__main__':
    main()
