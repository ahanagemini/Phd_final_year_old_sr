"""Usage:   dim_reduction.py --img_dir=img_dir
            dim_reduction.py --help | -help | -h

Perform dimension reduction of images in a directory
Arguments:
  img_dir   directory of training images for dimension reduction

Options:
  -h --help -h
"""

import time
from docopt import docopt
from pathlib import Path
import numpy as np
from scipy.fftpack import fft, dct
from sklearn.decomposition import IncrementalPCA
from skimage.transform import resize
import math
import csv

NUM_PIECES = 4

def preprocess_images(img_dir):
    '''
    Function to read the images in a directory and perform pre-pca processing
    Pre-pca processing includes reducing dimension to 128*128, 
    splitting to 4 64*64 images,
    using DCT to halve the dimension to 2^11
    Args: img_dir: image directory
    Returns: list of preprocessed images (each initial image represented by 4 dct pieces)
    '''
    pre_pca_list = []
    imglist = list(img_dir.rglob("*.npz"))
    imglist = sorted(imglist)
    read_time = 0.0
    dct_time = 0.0
    rescale_time = 0.0
    cut_time = 0.0
    for img_name in imglist:
        # read image
        start = time.time()
        image = np.load(img_name)
        image = image.f.arr_0
        read_time = read_time + time.time() - start
        # rescale to 128 * 128
        start = time.time()
        image = resize(image, (128, 128), order=3, preserve_range=True)
        rescale_time = rescale_time + time.time() - start
        cut_imgs = []
        cut_row_size = image.shape[0] // int(math.sqrt(NUM_PIECES))
        cut_col_size = image.shape[1] // int(math.sqrt(NUM_PIECES))
        # divide into 4 images, dct and then taking first half of DCT
        for i in range(int(math.sqrt(NUM_PIECES))):
            for j in range(int(math.sqrt(NUM_PIECES))):
                start = time.time()
                img_shard = image[i * cut_row_size: (i+1) * cut_row_size, j * cut_col_size: (j+1) * cut_col_size]
                cut_time = cut_time + time.time() - start
                start = time.time()
                img_shard = dct(img_shard.flatten(), norm='ortho')
                dct_time = dct_time + time.time() - start
                cut_imgs.append(img_shard[:img_shard.shape[0] // 2])
        pre_pca_list.extend(cut_imgs)

    img4list = [item for item in imglist for i in range(4)]
    print(f"Read_time {read_time}, Rescale time {rescale_time}, DCT_time {dct_time}, Cut time {cut_time}")
    return img4list, pre_pca_list

def perform_pca(pca_list):

    '''
    Function to perfrom PCA on a set of partial images
    Args: pca_list: list of images for prforming PCA on
    Returns: list of dimension reduced images
    '''
    start = time.time()
    transformer = IncrementalPCA(n_components=128, batch_size=200)
    X = np.array(pca_list)
    X_transformed = transformer.fit_transform(X)
    print(f"PCA time {time.time() - start}")
    return X_transformed.tolist()

def write_diversity(img_dir, pca_list, img4list):

    '''
    Function to write dimensio reduced images to file
    Args: 
        img_dir: image directory
        pca_list: the list to write to file
        im4list: file names for each item in the list
    '''
    [pca_list[i].insert(0, img4list[i]) for i in range(len(pca_list))]
    
    active_file = str(img_dir).split('/')[-1]
    with open(f"active_diversity_metrics/{active_file}.csv", 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerows(pca_list)



if __name__ == "__main__":
    args = docopt(__doc__)
    img_dir = Path(args["--img_dir"]).resolve()
    start = time.time()
    img4list, pre_pca_list = preprocess_images(img_dir)
    pca_list = perform_pca(pre_pca_list)
    write_diversity(img_dir, pca_list, img4list)
    print(f"Total time {time.time() - start}")

