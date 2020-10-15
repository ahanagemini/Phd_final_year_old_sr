from easydict import EasyDict as edict
import json
import os
config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 16 # [16] use 8 if your GPU memory is small, and use [2, 4] in tl.vis.save_images / use 16 for faster training
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9
cwd = os.getcwd()
## initialize G
config.TRAIN.num_patches = 1
config.TRAIN.n_epoch_init = 300
    # config.TRAIN.lr_decay_init = 0.1
    # config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)
config.TRAIN.img_path = f'/home/ahana/research/DWICutter/train/dwi'

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 2000
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)

## train set location
config.TRAIN.hr_img_path = f'/home/ahana/research/DWICutter/train/dwi/HR'
config.TRAIN.lr_img_path = f'/home/ahana/research/DWICutter/train/dwi/LR'

config.VALID = edict()
## test set location
config.VALID.hr_img_path = f'/home/ahana/research/DWICutter/valid/dwi/HR'
config.VALID.lr_img_path = f'/home/ahana/research/DWICutter/valid/dwi/LR'

config.TEST = edict()
config.TEST.hr_img_path = f'/home/ahana/research/DWICutter/test/dwi/HR'
config.TEST.lr_img_path = f'/home/ahana/research/DWICutter/test/dwi/LR'

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
