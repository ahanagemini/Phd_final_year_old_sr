from torch.utils.tensorboard import SummaryWriter

class Logger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)
    def scalar_summary(self, tag, value, step):
        '''took the summary logger from https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/logger.py'''
        self.writer.add_scalar(tag=tag, scalar_value=value, global_step=step)
        self.writer.flush()