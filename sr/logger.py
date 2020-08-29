import tensorflow as tf

class Logger(object):
    def __init__(self, log_dir):
        self.writer = tf.summary.create_file_writer(log_dir)
    def scalar_summary(self, tag, value, step):
        '''took the summary logger from https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/logger.py'''
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)
        self.writer.flush()