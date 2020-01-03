from argparse import ArgumentParser

import mmcv

from mmdet.core.evaluation import lvis_eval
from mmdet.datasets import build_dataset


def main():
    parser = ArgumentParser(description='LVIS Evaluation')
    parser.add_argument('config', help='config')
    parser.add_argument('result', help='result file path')
    parser.add_argument(
        '--types',
        type=str,
        nargs='+',
        choices=['proposal_fast', 'proposal', 'bbox', 'segm', 'keypoint'],
        default=['segm'],
        help='result types')
    parser.add_argument(
        '--max_dets',
        type=int,
        default=100,
        help='proposal numbers, only used for recall evaluation')
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.test)

    print('loading result file: {}'.format(args.result))
    result_files = dict()
    for eval_type in args.types:
        result_json_path = args.result + '.{}.json'.format(eval_type)
        mmcv.check_file_exist(result_json_path)
        result_files[eval_type] = result_json_path
    lvis_eval(result_files, args.types, dataset.lvis, args.max_dets)


if __name__ == '__main__':
    main()
