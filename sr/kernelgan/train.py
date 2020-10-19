import tqdm

from .data import DataGenerator
from .kernelGAN import KernelGAN
from .learner import Learner


# def train(conf, image, stats, X4=False):
def train(conf):
    gan = KernelGAN(conf)
    learner = Learner()
    data = DataGenerator(conf, gan)
    for iteration in tqdm.tqdm(range(conf.max_iters), ncols=60):
        [g_in, d_in, input_image, stat] = data.__getitem__(iteration)
        gan.train(g_in, d_in, input_image, stat)
        learner.update(iteration, gan)
    output_image = gan.finish()
    return output_image
