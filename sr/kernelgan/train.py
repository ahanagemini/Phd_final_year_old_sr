import tqdm

from kernelgan.configs import Config
from kernelgan.data import DataGenerator
from kernelgan.kernelGAN import KernelGAN
from kernelgan.learner import Learner


def train(image, stats, X4=False):
    conf = Config().parse()
    gan = KernelGAN(conf, X4)
    learner = Learner()
    data = DataGenerator(conf, image, stats, gan)
    for iteration in tqdm.tqdm(range(conf.max_iters), ncols=60):
        [g_in, d_in, input_image, stat] = data.__getitem__(iteration)
        gan.train(g_in, d_in, input_image, stat)
        learner.update(iteration, gan)
    output_image = gan.finish()
    return output_image

