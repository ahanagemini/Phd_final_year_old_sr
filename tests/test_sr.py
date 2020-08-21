from sr import __version__
import unittest
from PIL import Image
import numpy as np
from pathlib import Path

def test_version():
    assert __version__ == '0.1.0'

from sr import cutter

# TODO: Write a test here for using cutter
class TestLoader(unittest.TestCase):
    def test_load(self):
        path = Path(r"/home/venkat/Documents/PiyushKumarProject/sr/tests/TestFiles/")
        files = sorted(path.rglob("*.png"))
        for file in files:
            self.assertTrue((cutter.Loader(file)).all() == (np.array(Image.open(file))).all())

if __name__ == '__main__':
        unittest.main()
