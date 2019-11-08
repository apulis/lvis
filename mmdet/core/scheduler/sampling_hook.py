import numpy as np

from .base_scheduler import BaseSchedulerHook


class RepeatFactorSamplingHook(BaseSchedulerHook):

    def __init__(self, data_loader, thres=0.001, interpolate=False, **kwargs):
        super(RepeatFactorSamplingHook, self).__init__(**kwargs)
        assert data_loader.dataset.lvis is not None, \
            'RepeatFactorSamplingHook only supports for LVIS dataset.'
        self.data_loader = data_loader
        self.dataset = data_loader.dataset
        self.thres = thres
        self.interpolate = interpolate
        self.img_repeat_factors = self.compute_img_repeat_factors()

    def compute_cat_repeat_factors(self):
        dataset = self.dataset
        cats = dataset.lvis.load_cats(dataset.cat_ids)
        cat_freqs = np.zeros(len(cats))
        for i, cat in enumerate(cats):
            cat_freqs[i] = cat['image_count'] / len(dataset.img_ids)
        cat_repeat_factors = np.maximum(1, np.sqrt(self.thres / cat_freqs))
        return cat_repeat_factors

    def compute_img_repeat_factors(self):
        cat_repeat_factors = self.compute_cat_repeat_factors()
        img_repeat_factors = []
        for i, img_info in enumerate(self.dataset.img_infos):
            img_id = img_info['id']
            ann_ids = self.dataset.lvis.get_ann_ids(img_ids=[img_id])
            anns = self.dataset.lvis.load_anns(ids=ann_ids)

            cats_in_img = []
            for ann in anns:
                cats_in_img.append(ann['category_id'])

            cat_repeat_factors_in_img = []
            for unique_cat_id in set(cats_in_img):
                cat_repeat_factors_in_img.append(
                    cat_repeat_factors[unique_cat_id - 1])
            img_repeat_factors.append(np.max(cat_repeat_factors_in_img))
        return np.asarray(img_repeat_factors)

    def apply_curriculum(self, runner):
        step = runner.epoch / (runner.max_epochs-1)
        if self.interpolate:
            # interpolate img repeat factors
            # concave -> change slowly. convex -> change fast
            repeat_factors = (1 - step) + step * self.img_repeat_factors
            repeat_factors = np.round(repeat_factors).astype(np.int)
        else:
            repeat_factors = np.round(self.img_repeat_factors).astype(np.int)

        repeated_img_indices = []
        for idx, repeat_num in enumerate(repeat_factors):
            repeated_img_indices.extend(np.repeat(idx, repeat_num).tolist())

        sampler = self.data_loader.sampler
        sampler.set_repeated_indices(repeated_img_indices)
