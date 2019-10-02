import numpy as np 
import torch
import torch.distributed as dist

from .base_scheduler import BaseSchedulerHook


class RepeatFactorSamplingHook(BaseSchedulerHook): 

    def __init__(self, data_loader, thres=0.001, **kwargs): 
        super(RepeatFactorSamplingHook, self).__init__(**kwargs)
        assert data_loader.dataset.lvis is not None, \
            'RepeatFactorSamplingHook only supports for LVIS dataset.'
        self.data_loader = data_loader
        self.dataset = data_loader.dataset 
        self.thres = thres 

    def _get_cat_level_repeat_factor(self, thres): 
        categories = self.dataset.categories

        for i, cat in enumerate(categories): 
            cat['category_level_repeat_factor'] = \
                np.max(
                    (1, np.sqrt(thres/cat['category_freq']))
                )
        return categories

    def _get_image_level_repeat_factor(self, categories): 
        img_infos = self.dataset.img_infos 

        repeat_factors = []
        for i in range(len(img_infos)): 
            img_info = img_infos[i] 

            ann_ids = self.dataset.lvis.get_ann_ids(
                img_ids=[img_info['id']])
            anns = self.dataset.lvis.load_anns(
                ids=ann_ids)

            categories_in_img = []
            for ann in anns: 
                cat_id = ann['category_id']
                categories_in_img.append(cat_id)

            cat_level_repeat_factor_in_img = []
            for unique_cat_id in set(categories_in_img):
                cat = categories[unique_cat_id-1]
                cat_level_repeat_factor_in_img.append(
                    cat['category_level_repeat_factor'])

            img_level_repeat_factor = np.max(
                cat_level_repeat_factor_in_img)
            repeat_factors.append(img_level_repeat_factor)

        return repeat_factors

    def apply_curriculum(self, runner):
        step = runner.epoch
        total = runner.max_epochs-1
        curriculum_factor = self.curriculum_func(step, total)
        thres = self.thres*curriculum_factor
        runner.logger.info(
            'current phase: {}/{}={:.3f}\t'.format(
                step, total, step/total) +
            'curriculum factor set to {:.4f}, '.format(
                curriculum_factor) +
            'and repeat factor thres set to {:.4f}'.format(
                thres))

        categories = self._get_cat_level_repeat_factor(thres)
        repeat_factor = self._get_image_level_repeat_factor(
            categories)

        sampler = self.data_loader.sampler 
        sampler.set_repeat_factors(repeat_factor)