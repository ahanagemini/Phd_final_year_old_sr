
import numpy as np
import os
import tifffile
import nibabel as nib
import json
import argparse
import sys
from pathlib import Path
from PIL import Image
from stat_plotter import PlotStat
from tqdm import tqdm
from loadmap import ColorMap
import matplotlib.pyplot as plt



class Converter:
    """This class takes a image and converts it into an interpolated form"""

    def loader(self, ifile):
        """


        Parameters
        ----------
        file : TYPE
            DESCRIPTION.

        Returns
        -------
        imageArray : TYPE
            DESCRIPTION.

        """

        ImagePaths = [".jpg", ".png", ".jpeg", ".gif"]
        ImageArrayPaths = [".npy", ".npz"]
        TiffPaths = [".tiff", ".tif"]
        niftiPaths = [".nii", ".nii.gz"]
        fname = str(ifile.name).lower()
        fileExt = "." + ".".join(fname.split(".")[1:])
        if fileExt in ImagePaths:
            image = Image.open(ifile)
            image = np.array(image.convert(mode="L"))
        if fileExt in TiffPaths:
            # 16 bit tifffiles are not read correctly by Pillow
            image = tifffile.imread(str(ifile))
        if fileExt == ".npz":
            image = np.load(ifile)
            image = image.f.arr_0  # Load data from inside file.
        elif fileExt in ImageArrayPaths:
            image = np.load(ifile)
        elif fileExt in niftiPaths:
            img = nib.load(ifile)
            image = img.get_data()

        return image

    def image_clipper(self, image, stats):
        """
        This method clips the data max value and min value to 5 % of max value and min value of entire image distribution
        in stats file
        Parameters
        ----------
        image: image matrix
        stats: image statistics

        Returns
        -------

        """
        max_value = 0.05 * stats["max"]
        min_value = 0.05 * stats["min"]
        width, height = image.shape
        for x in range(width):
            for y in range(height):
                if image[x][y] > max_value:
                    image[x][y] = max_value
                elif image[x][y] < min_value:
                    image[x][y] = min_value

        return image
    def image_converter(self, idir, odir):
        """This method will take an input directory and load all the images present in the directory and interpolates
        them and finally stores them in a output directory"""
        plotter = PlotStat()
        idir = Path(idir)
        odir = Path(odir)

        if not os.path.isdir(odir):
            os.makedirs(odir)

        image_paths = idir.rglob("*")
        stats = json.load(open(str(idir/"stats.json")))
        max_value = 0.05 * stats["max"]
        min_value = 0.05 * stats["min"]
        lab = ColorMap("lab.yaml", min_value, max_value)
        for i, image_path in enumerate(tqdm(image_paths)):
            image_name = os.path.splitext(image_path.name)[0]
            image_matrix = self.loader(image_path)
            image_matrix = self.image_clipper(image_matrix, stats)
            #image_matrix = plotter.t_interpolate(image_matrix, "bicubic", 0.25)

            # converting each image pixel to color
            width, height = image_matrix.shape
            final_matrix = np.zeros((width, height, 3), dtype=np.uint8)
            for x in range(width):
                for y in range(height):
                     final_matrix[x][y] = lab.convert(image_matrix[x][y])

            image = Image.fromarray(final_matrix)
            image.save(str(odir/ image_name)+r".png")

class Configurator:
    """this class is for configurating input and output"""
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--input_dir", default=os.path.dirname(os.path.abspath(__file__))+r"/input_images",
                                 help="use this command to pass the input directory where the images are present")
        self.parser.add_argument("--output_dir", default=os.path.dirname(os.path.abspath(__file__))+r"/output_images",
                                 help="use this command to pass the out put directory where you want images to be saved")

    def parse(self, args=None):
        """Parse the configuration"""
        self.conf = self.parser.parse_args(args=args)
        return self.conf


if __name__ == "__main__":
    '''
    if (
        len(sys.argv) == 1
        or "--input_dir_path" not in str(sys.argv)
        or "--output" not in str(sys.argv)
    ):
        sys.argv.append("-h")
    conf = Configurator().parse()
    '''
    idir = r"/home/venkat/Documents/PiyushKumarProject/earth1"
    odir = r"/home/venkat/Documents/PiyushKumarProject/earth_1_color"
    colormap = Converter()
    colormap.image_converter(idir, odir)