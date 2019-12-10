import itertools

import mmcv
import numpy as np
from terminaltables import AsciiTable

from mmdet.lvis import LVISEval


def lvis_eval(result_files, result_types, lvis, max_dets=100):
    if result_files is None:
        print('Nothing to evaluate.')
        return

    for res_type in result_types:
        if res_type not in ['bbox', 'segm']:
            raise KeyError(
                'invalid iou_type: {} for evaluation'.format(res_type))
        try:
            mmcv.check_file_exist(result_files[res_type])
        except IndexError:
            print('No prediction found.')
            break
        lvis_eval = LVISEval(lvis, result_files[res_type], iou_type=res_type)
        lvis_eval.params.max_dets = max_dets
        lvis_eval.run()
        lvis_eval.print_results()

        # Compute per-category AP
        # from https://github.com/facebookresearch/detectron2/blob/03064eb5bafe4a3e5750cc7a16672daf5afe8435/detectron2/evaluation/coco_evaluation.py#L259-L283 # noqa
        if True:
            precisions = lvis_eval.eval['precision']
            catIds = lvis.get_cat_ids()
            # precision has dims (iou, recall, cls, area range, max dets)
            assert len(catIds) == precisions.shape[2]

            results_per_category = []
            for idx, catId in enumerate(catIds):
                # area range index 0: all area ranges
                # max dets index -1: typically 100 per image
                nm = lvis.load_cats([catId])[0]
                precision = precisions[:, :, idx, :]
                precision = precision[precision > -1]
                ap = np.mean(precision) if precision.size else float('nan')
                results_per_category.append(
                    ('{}'.format(nm['name']),
                     '{:0.3f}'.format(float(ap * 100))))

            N_COLS = min(6, len(results_per_category) * 2)
            results_flatten = list(itertools.chain(*results_per_category))
            headers = ['category', 'AP'] * (N_COLS // 2)
            results_2d = itertools.zip_longest(
                *[results_flatten[i::N_COLS] for i in range(N_COLS)])
            table_data = [headers]
            table_data += [result for result in results_2d]
            table = AsciiTable(table_data)
            print(table.table)
            mmcv.dump(results_per_category, 'interp_classwise_ap.pkl')
