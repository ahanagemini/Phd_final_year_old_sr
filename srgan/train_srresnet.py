#! /usr/bin/python
# -*- coding: utf8 -*-

import os
import time
import random
import numpy as np
import scipy, multiprocessing
import tensorflow as tf
import tensorlayer as tl
from model import get_G, get_D
from config import config
import tifffile
import json
from skimage import metrics

## Adam
batch_size = config.TRAIN.batch_size  # use 8 if your GPU memory is small, and change [4, 4] in tl.vis.save_images to [2, 4]
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every
shuffle_buffer_size = 128
num_patches = config.TRAIN.num_patches
save_dir = "results"
tl.files.exists_or_mkdir(save_dir)
checkpoint_dir = "models"
tl.files.exists_or_mkdir(checkpoint_dir)
f = open(config.TRAIN.lr_img_path + '/stats.json')
stats = json.load(f)
max_val = float(stats["max"])

def get_train_data():
    # load dataset
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.npz', printable=False))#[0:20]
    train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.npz', printable=False))

    ## If your machine have enough memory, please pre-load the entire train set.
    train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    train_lr_imgs = tl.vis.read_images(train_lr_img_list, path=config.TRAIN.lr_img_path, n_threads=32)
    # dataset API and augmentation

    def generator_train():
        for index, hr in enumerate(train_hr_imgs):
            lr = train_lr_imgs[index]
            hr = tf.expand_dims(hr, 2)
            lr = tf.expand_dims(lr, 2)
            for i in range(num_patches):
                img = tf.concat([hr, lr], 2)
                yield img
    def _map_fn_train(img):
        patch = tf.image.random_crop(img, [64, 64, 2])
        patch = patch / (max_val / 2.)
        patch = patch - 1.
        patch = tf.image.random_flip_left_right(patch)
        hr_patch = tf.slice(patch, [0, 0, 0], [64, 64, 1])
        lr_patch = tf.slice(patch, [0, 0, 1], [64, 64, 1])
        return lr_patch, hr_patch
    train_ds = tf.data.Dataset.from_generator(generator_train, output_types=(tf.float32))
    train_ds = train_ds.map(_map_fn_train, num_parallel_calls=multiprocessing.cpu_count())
    train_ds = train_ds.shuffle(shuffle_buffer_size)
    train_ds = train_ds.prefetch(buffer_size=2)
    train_ds = train_ds.batch(batch_size)
    return train_ds, len(train_hr_imgs)

def train():
    G = get_G((batch_size, 64, 64, 1))
    f = open(config.TRAIN.lr_img_path + '/stats.json')
    stats = json.load(f)

    lr_v = tf.Variable(lr_init)
    g_optimizer_init = tf.optimizers.Adam(lr_v, beta_1=beta1)

    G.train()

    train_ds, num_train_imgs = get_train_data()
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.TEST.hr_img_path, regx='.*.npz', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.TEST.lr_img_path, regx='.*.npz', printable=False))

    valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.TEST.lr_img_path, n_threads=32)
    valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.TEST.hr_img_path, n_threads=32)
    best_psnr = 0.0
    ## initialize learning (G)
    n_step_epoch = round((num_patches * num_train_imgs) // batch_size)
    for epoch in range(n_epoch_init):
        G.train()
        step_time = time.time()
        for step, (lr_patchs, hr_patchs) in enumerate(train_ds):
            if lr_patchs.shape[0] != batch_size: # if the remaining data in this epoch < batch_size
                break
            with tf.GradientTape() as tape:
                fake_hr_patchs = G(lr_patchs)
                mse_loss = tl.cost.mean_squared_error(fake_hr_patchs, hr_patchs, is_mean=True)
            grad = tape.gradient(mse_loss, G.trainable_weights)
            g_optimizer_init.apply_gradients(zip(grad, G.trainable_weights))
            if step % 100 == 0:
                print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, mse: {:.3f} ".format(
                      epoch, n_epoch_init, step, n_step_epoch, time.time() - step_time, mse_loss))
                step_time = time.time()
        if (epoch != 0) and (epoch % 10 == 0):
            G.save_weights(os.path.join(checkpoint_dir, f'srresnet_{epoch}.h5'))
        psnr = evaluate(valid_hr_img_list, valid_lr_img_list, G)
        if psnr > best_psnr:
            best_psnr = psnr
            G.save_weights(os.path.join(checkpoint_dir, f'srresnet_best.h5'))

    G.save_weights(os.path.join(checkpoint_dir, f'srresnet_final.h5'))

def evaluate(test_hr_img_list=None, test_lr_img_list=None, G=None, save_img=False):
    if test_hr_img_list is None:
        test_hr_img_list = sorted(tl.files.load_file_list(path=config.TEST.hr_img_path, regx='.*.npz', printable=False))
        test_lr_img_list = sorted(tl.files.load_file_list(path=config.TEST.lr_img_path, regx='.*.npz', printable=False))

    test_lr_imgs = tl.vis.read_images(test_lr_img_list, path=config.TEST.lr_img_path, n_threads=32)
    test_hr_imgs = tl.vis.read_images(test_hr_img_list, path=config.TEST.hr_img_path, n_threads=32)

    if G is None:
        G = get_G([1, None, None, 1])
        G.load_weights(os.path.join(checkpoint_dir, 'srresnet_final.h5'))
    G.eval()
    tot_psnr = 0.0
    tot_ssim = 0.0
    for i in range(len(test_lr_imgs)):
        test_lr_img = test_lr_imgs[i]
        test_hr_img = test_hr_imgs[i]
        test_lr_img = tf.expand_dims(test_lr_img, 2)
        test_hr_img = tf.expand_dims(test_hr_img, 2)
        tl.vis.save_image(np.clip(test_lr_img, 0, max_val).astype(np.uint16), os.path.join(save_dir, f'test_{i}_lr.tiff'))
        size = [test_lr_img.shape[0], test_lr_img.shape[1]]
        test_lr_img = np.asarray(test_lr_img, dtype=np.float32)
        test_lr_img = (test_lr_img / (max_val /2.)) - 1
        test_lr_img = test_lr_img[np.newaxis,:,:,:]
        out = G(test_lr_img).numpy()
        # print("LR size: %s /  generated HR size: %s" % (test_lr_img.shape, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
        # print("[*] save images")
        sr_img = out[0] + 1
        sr_img = sr_img * (max_val / 2.)
        sr_img = np.clip(sr_img, 0, max_val)
        sr_img = sr_img.astype(np.uint16)
        size = sr_img.shape[1]
        if save_img:
            tl.vis.save_image(sr_img, os.path.join(save_dir, f'test_{i}_sr.tiff'))
            tl.vis.save_image(test_hr_img, os.path.join(save_dir, f'test_{i}_hr.tiff'))
        psnr = metrics.peak_signal_noise_ratio(test_hr_img.numpy().reshape(size, -1).astype(np.float32),
                                               sr_img.reshape(size, -1).astype(np.float32), data_range=max_val)
        ssim = metrics.structural_similarity(test_hr_img.numpy().reshape(size, -1).astype(np.float32),
                                             sr_img.reshape(size, -1).astype(np.float32),
                                             gaussian_weights=True, data_range=max_val)
        tot_psnr = tot_psnr + psnr
        tot_ssim = tot_ssim + ssim
        
    print("\nAvg PSNR: {:.6f}".format(tot_psnr / len(test_lr_imgs)))
    print("\nAvg SSIM: {:.6f}".format(tot_ssim / len(test_lr_imgs)))
    return tot_psnr

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='srgan', help='srgan, evaluate')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'srgan':
        train()
        evaluate(save_img=True)
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate()
    else:
        raise Exception("Unknow --mode")

