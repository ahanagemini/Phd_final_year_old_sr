import unittest
import numpy as np
from .data import Normalize
from sr.configs import Config
import random

class MyTestCase(unittest.TestCase):
    def normalize_test(self):
        '''This test is done to check if the normalizing function is working correctly'''
        image_array = np.random.rand(10, 10)
        stat = {}
        stat["mean"] = np.mean(image_array)
        stat["std"] = np.std(image_array)
        norm_image = (image_array - stat["mean"])/stat["std"]
        kernalgan_norm = Normalize(image=image_array, stat=stat)
        self.assertEqual(norm_image, kernalgan_norm, "The norms are not same")

    def kernelGanTest(self):
        '''
        this test is done to check whether kernelGan is giving a correct image
        '''

        image_array = np.random.rand(10,10,1)
        conf = Config.parse()
        
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
