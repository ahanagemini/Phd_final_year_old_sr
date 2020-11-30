from sr import __version__
import unittest
import numpy as np
import os
from pathlib import Path

def test_version():
    assert __version__ == '0.1.0'

from sr import cutter

# TODO: Write a test here for using cutter
class TestLoader(unittest.TestCase):
    def test_ssim(self):


    def test_load(self):
        matrix = np.random.rand(256, 256)
        np.savez(os.getcwd()+"/TestFiles/matrix.npz", matrix)

        matrix_file = Path(os.getcwd()+"/TestFiles/matrix.npz")
        imatrix = np.load(matrix_file)
        imatrix = imatrix.f.arr_0
        self.assertTrue((cutter.loader(matrix_file)).all() == imatrix.all())

    def test_file(self):
        input_extract = ImportantFunctions()
        matrix = np.random.rand(256, 256)
        np.savez(os.getcwd()+"/TestFiles/matrix_2.npz", matrix)
        img = Path(os.getcwd()+"/TestFiles/Dog/Dog.png")
        imatrix_1 = Path(os.getcwd()+"/TestFiles/matrix/matrix.npz")
        imatrix_2 = Path(os.getcwd()+"/TestFiles/matrix_2/matrix_2.npz")


        file_paths = [img, imatrix_1, imatrix_2]
        file_paths = sorted(file_paths)
        input_file = Path(os.getcwd()+"/TestFiles/")
        output_file = Path(os.getcwd()+"/TestFiles/")
        input_path= cutter.scan_idir(input_file, output_file)
        input_path = sorted(input_extract.inputFileExtractorFromTuple(input_path))
        self.assertListEqual(input_path, file_paths)


class ImportantFunctions:
    def inputFileExtractorFromTuple(self, list_1):
        input_files = []
        for i in range(len(list_1)):
            input_file, output_file = list_1[i]
            input_files.append(input_file)
        return input_files

if __name__ == '__main__':
        unittest.main()
