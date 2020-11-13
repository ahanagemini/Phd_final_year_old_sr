import unittest
import random
import numpy as np
import copy
from pathlib import Path
from scipy.spatial.distance import directed_hausdorff
from dim_reduction import preprocess_images, perform_pca

class TestDimReduction(unittest.TestCase):
'''
Unit test for dim_reduction.py. 
The min L1 loss between the 10 most dissimilar images and current image
should be greater than max loss between 10 most similar images and current image
'''
    def test_dim_reduction(self):
        """
        Function to perform unit test
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
            indices = np.argpartition(d_u_matrix[i], -10)[-10:]
            indices_l = np.argpartition(d_u_matrix[i], 11)[:11]
            cur_list.append(files[i])
            for x in indices:
                cur_list.append(files[x])
            for y in indices_l:
                if y != i:
                    cur_list.append(files[y])
            sim_dissim_list.append(cur_list)
        count = 0
        # compute L1 error for cur image and rest and check for anomaly
        for files_list in sim_dissim_list:
            image = np.load(files_list[0])
            image = image.f.arr_0
            maxim = 0
            minim = float('inf')
            for i in range(1, 11):
                image1 = np.load(files_list[i])
                image1 = image1.f.arr_0
                loss = np.mean(np.abs(image - image1))
                if loss < minim:
                    minim = loss

            for i in range(11, 21):
                image1 = np.load(files_list[i])
                image1 = image1.f.arr_0
                loss = np.mean(np.abs(image - image1))
                if loss > maxim:
                    maxim = loss
            if minim < maxim:
                print(minim, maxim, files_list[0])
                count = count + 1

            self.assertEqual(count, 0, "Should be 0")

if __name__ == "__main__":
    unittest.main()





