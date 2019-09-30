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

    def _get_cat_level_repeat_factor(self): 
        categories = self.dataset.categories

        for i, cat in enumerate(categories): 
            cat['category_level_repeat_factor'] = \
                np.max(
                    (1, np.sqrt(self.thres/cat['category_freq']))
                )
        return categories

    def _get_image_level_repeat_factor(self, categories, factor): 
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
                cat_level_repeat_factor_in_img) ** factor
            repeat_factors.append(img_level_repeat_factor)

        return repeat_factors

    def apply_curriculum(self, runner):
        phase = runner.epoch/(runner.max_epochs-1)
        if not self.reverse: 
            phase = 1 - phase
        curriculum_factor = self.curriculum_func(phase)
        categories = self._get_cat_level_repeat_factor()
        repeat_factor = self._get_image_level_repeat_factor(
            categories, curriculum_factor)

        sampler = self.data_loader.sampler 
        sampler.set_repeat_factors(repeat_factor)