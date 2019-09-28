from __future__ import division
import argparse
import os

import torch
from mmcv import Config

from mmdet import __version__
from mmdet.apis import (get_root_logger, init_dist, set_random_seed,
                        train_detector)
from mmdet.datasets import build_dataset
from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    #model = build_detector(
    #    cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        datasets.append(build_dataset(cfg.data.val))
    # add an attribute for visualization convenience
    #model.CLASSES = datasets[0].CLASSES
    import pdb 
    pdb.set_trace()

if __name__ == '__main__':
    main()
