import torch
import torch.distributed as dist
from mmcv.runner import Hook

from .functions import linear, constant, convex, composite, concave


class BaseSchedulerHook(Hook):
    
    def __init__(self, 
                 interval,
                 schedule='linear',
                 reverse=True):
        self.interval = interval
        if schedule not in ['linear', 'constant']: 
            raise KeyError('{} is not supported for curriculum function.')
        self.schedule = schedule 
        self.reverse = reverse 
        self.curriculum_func = eval(self.schedule)

    def before_run(self, runner):
        pass 

    def before_train_epoch(self, runner): 
        if not self.every_n_epochs(runner, self.interval): 
            return 
        self.apply_curriculum(runner)
        dist.barrier() 

    def apply_curriculum(self): 
        raise NotImplementedError

