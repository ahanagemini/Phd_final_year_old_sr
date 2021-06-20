import random
import numpy as np
import copy
from pathlib import Path
import os
from PIL import Image
import tifffile
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff
from dim_reduction import preprocess_images, perform_pca

def create_sim_dissim_list():
    """
    Function to compute 4 most similar and dissimilar images to image
    Returns: List of 100 images and their 4 most similar and dissimilar images
    """
    # perform dimension reduction
    img4list, pre_pca_list = preprocess_images(Path("data/t1ce").resolve())
    files = [str(img4list[i]) for i in range(len(img4list)) if i % 4 == 0]
    pca_list = perform_pca(pre_pca_list)
    # only use first 200 images to save time. each image is 4 points
    diversity_values = pca_list[:800]
    ind = random.sample(range(0, len(diversity_values) // 4 - 1), 100)
    d_u_matrix = np.zeros((len(diversity_values) // 4, len(diversity_values) // 4))
    # compute diversity matrix
    for i in range(len(diversity_values) // 4):
        for j in range(i + 1, len(diversity_values) // 4):
            u = np.array(diversity_values[4*i: 4*(i+1)])
            v = np.array(diversity_values[4*j: 4*(j+1)])
            temp = max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
            d_u_matrix[i][j] = temp
            d_u_matrix[j][i] = temp

    # identify 10 most similar and 10 dissimilar images from reduced dimension
    sim_dissim_list = []
    dummy_list = []
    for i in ind:
        cur_list = copy.deepcopy(dummy_list)
        indices = np.argpartition(d_u_matrix[i], -4)[-4:]
        indices_l = np.argpartition(d_u_matrix[i], 5)[:5]
        cur_list.append(files[i])
        for x in indices:
            cur_list.append(files[x])
        for y in indices_l:
            if y != i:
                cur_list.append(files[y])
        sim_dissim_list.append(cur_list)
    return sim_dissim_list

def save_images(sim_dissim_list):
    """
    Function to create and save images for visualizing dim_reductio.py quality
    Args: sim_dissim_list
    """
    print(len(sim_dissim_list), len(sim_dissim_list[0]))
    for files_list in sim_dissim_list:
        image = np.load(files_list[0])
        image = image.f.arr_0
        image = np.load(files_list[1])
        image = image.f.arr_0
        image1 = np.load(files_list[2])
        image1 = image1.f.arr_0
        image2 = np.load(files_list[3])
        image2 = image2.f.arr_0
        image3 = np.load(files_list[4])
        image3 = image3.f.arr_0
        img_x = np.vstack((image, image1))
        img_y = np.vstack((image2, image3))
        image = np.load(files_list[5])
        image = image.f.arr_0
        image1 = np.load(files_list[6])
        image1 = image1.f.arr_0
        image2 = np.load(files_list[7])
        image2 = image2.f.arr_0
        image3 = np.load(files_list[8])
        image3 = image3.f.arr_0
        img_p = np.vstack((image, image1))
        img_q = np.vstack((image2, image3))
        image = np.load(files_list[0])
        image = image.f.arr_0
        img = np.vstack((image, image))
        if np.mean(img_x) > np.mean(img_p):
            img_p = img_p * 4
            img_q = img_q * 4
        elif np.mean(img_p) > np.mean(img_x):
            img_x = img_x * 4
            img_x = img_y * 4
        if np.mean(img_x) > np.mean(img):
            img = img * 4
        img_save = np.hstack((img_x, img_y, img, img_p, img_q))
        filename = 'cmp_red_dim/' + (str(files_list[0]).split('/')[-1]).split('.')[0] + '.png'
        plt.imsave(filename, img_save, cmap="gray")

if __name__ == "__main__":
    sim_dissim_list = create_sim_dissim_list()
    save_images(sim_dissim_list)





