import mmcv
import numpy as np

from mmdet.apis.lvis import LVIS, LVISEval 


def lvis_eval(result_files, result_types, lvis, max_dets=100): 
    if result_files is None: 
        print('Nothing to evaluate.') 
        return 

    for res_type in result_types: 
        if res_type not in ['bbox', 'segm']: 
            raise KeyError('invalid iou_type: {} for evaluation'.format(
                res_type))
        try:
            mmcv.check_file_exist(result_files[res_type])
        except IndexError:
            print('No prediction found.')
            break
        lvis_eval = LVISEval(
            lvis,
            result_files[res_type],
            iou_type=res_type)
        lvis_eval.params.max_dets = max_dets 
        lvis_eval.run() 
        lvis_eval.print_results()