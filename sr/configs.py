import argparse
import torch
import os


# noinspection PyPep8
class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.conf = None
        self.image = None
        self.stats = None

        # Paths
        self.parser.add_argument('--img_name', default='image1', help='image name for saving purposes')
        self.parser.add_argument('--input_dir_path', default=os.path.dirname(__file__) + '/input',
                                 help='all inputs are in this path')
        self.parser.add_argument('--cutting_output_dir_path', default=os.path.dirname(__file__) + '/cutter_out',
                                 help="here cut images will be stored")
        self.parser.add_argument('--output_dir_path', default=os.path.dirname(__file__) + '/results', help='results path')
        self.parser.add_argument('--stat_image_path', default=os.path.dirname(__file__) + '/training_data/stats.json', help='path to the stat file of image')

        # Sizes
        self.parser.add_argument('--input_crop_size', type=int, default=64, help='Generators crop size')
        self.parser.add_argument('--scale_factor', type=float, default=0.5, help='The downscaling scale factor')
        self.parser.add_argument('--X4', action='store_true', help='The wanted SR scale factor')

        # Network architecture
        self.parser.add_argument('--G_chan', type=int, default=64, help='# of channels in hidden layer in the G')
        self.parser.add_argument('--D_chan', type=int, default=64, help='# of channels in hidden layer in the D')
        self.parser.add_argument('--G_kernel_size', type=int, default=13, help='The kernel size G is estimating')
        self.parser.add_argument('--D_n_layers', type=int, default=7, help='Discriminators depth')
        self.parser.add_argument('--D_kernel_size', type=int, default=7, help='Discriminators convolution kernels size')

        # Iterations
        self.parser.add_argument('--max_iters', type=int, default=3000, help='# of iterations')

        # Optimization hyper-parameters
        self.parser.add_argument('--g_lr', type=float, default=2e-4, help='initial learning rate for generator')
        self.parser.add_argument('--d_lr', type=float, default=2e-4, help='initial learning rate for discriminator')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='Adam momentum')

        # Trainer.py Parameters
        self.parser.add_argument('--train', default=os.path.dirname(__file__) + r"/results/train",
                                 help="train files path")
        self.parser.add_argument('--valid', default=os.path.dirname(__file__) + r"/results/valid",
                                 help="valid files path")
        self.parser.add_argument('--log_dir', default=os.path.dirname(__file__) + '/logger',
                                 help="the log files will be stored in this directory")
        self.parser.add_argument('--architecture', default="edsr_8_256", help="give the model to be train")
        self.parser.add_argument('--num_epochs', type=int, default=100, help="the total number of epochs")
        self.parser.add_argument('--lognorm', type=bool, default=False, help="check whether lognorm is required or not")
        self.parser.add_argument('--debug_pics', type=bool, default=False, help="check if debug pics are required")
        self.parser.add_argument('--aspp', type=bool, default=False, help="check if edsr needs aspp")
        self.parser.add_argument('--dilation', type=bool, default=False, help="check if edsr needs dilation")
        self.parser.add_argument('--act', default="leakyrelu", help="activation type relu or leakyrelu for edsr")
        self.parser.add_argument('--kernel_factor', default="--X4",
                                 help="the type of downscaling in training X2, X4, X8")
        self.parser.add_argument('--model_save', default=os.path.dirname(__file__)+"/saved_models",
                                 help= "the path where model will be saved")

        # Tester.py Parameters
        self.parser.add_argument("--test_input_dir")
        self.parser.add_argument('--active', type=bool, default=False,
                                 help="Whether to save per-image metrics for active learning selection")
        self.parser.add_argument('--save_slice', type=bool, default=False,
                                 help="If we want to save the slice as a png image")




        # GPU
        self.parser.add_argument('--gpu_id', type=int, default=0, help='gpu id number')

        # Kernel post processing
        self.parser.add_argument('--n_filtering', type=float, default=40, help='Filtering small values of the kernel')

        # Number of resize operations
        self.parser.add_argument('--n_resize', type=int, default=10, help='Number of resize operations/scales.')
        self.parser.add_argument('--noise_scale', type=float, default=1.,
                                 help='ZSSR uses this to partially de-noise images')
        self.parser.add_argument('--real_image', action='store_true', help='ZSSRs configuration is for real images')


    def parse(self, args=None):
        """Parse the configuration"""
        self.conf = self.parser.parse_args(args=args)
        self.set_gpu_device()
        self.conf.G_structure = [7, 5, 3, 1, 1, 1]
        return self.conf

    def set_gpu_device(self):
        """Sets the GPU device if one is given"""
        if os.environ.get('CUDA_VISIBLE_DEVICES', '') == '':
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.conf.gpu_id)
            torch.cuda.set_device(0)
        else:
            torch.cuda.set_device(self.conf.gpu_id)


