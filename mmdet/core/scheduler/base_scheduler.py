from functools import partial

import torch
import torch.distributed as dist
from mmcv.runner import Hook

from .functions import (linear, constant, convex, 
                        composite, concave, quadratic)


class BaseSchedulerHook(Hook):
    
    def __init__(self, 
                 interval,
                 schedule='linear',
                 self_learning=None):
        self.interval = interval
        if schedule not in ['linear', 'constant', 'convex', 
            'composite', 'concave', 'quadratic']: 
            raise KeyError('{} is not supported for curriculum function.')
        self.schedule = schedule 

        curriculum_func = eval(self.schedule)
        self.curriculum_func = partial(curriculum_func, **self_learning) \
            if self_learning else curriculum_func

    def before_run(self, runner):
        pass 

    def before_train_epoch(self, runner): 
        if not self.every_n_epochs(runner, self.interval): 
            return 
        self.apply_curriculum(runner)
        dist.barrier() 

    def apply_curriculum(self): 
        raise NotImplementedError

