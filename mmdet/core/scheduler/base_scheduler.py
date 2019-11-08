import torch.distributed as dist
from mmcv.runner import Hook


class BaseSchedulerHook(Hook):

    def __init__(self, interval):
        if interval is None:
            interval = 1
        self.interval = interval

    def before_run(self, runner):
        pass

    def before_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        self.apply_curriculum(runner)
        dist.barrier()

    def apply_curriculum(self):
        raise NotImplementedError
