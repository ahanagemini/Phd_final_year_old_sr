from torch.utils.tensorboard import SummaryWriter
import shutil
import os


class Logger(object):
    def __init__(self, log_dir):
        if os.path.isdir(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)

    def scalar_summary(self, tag, value, step):
        """took the summary logger from https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/logger.py"""
        self.writer.add_scalar(tag=tag, scalar_value=value, global_step=step)
        self.writer.flush()

    def model_graph(self, model, input_to_model):
        """This method is used to draw the model"""
        self.writer.add_graph(model, input_to_model)
        self.writer.close()
