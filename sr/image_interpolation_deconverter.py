
import numpy as np
import os
import tifffile
import nibabel as nib
import json
import argparse
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from loadmap import ColorMap
import matplotlib.pyplot as plt

class Deconverter:
    """This class takes a image and deconverts it back to floating point and calculates ssim between the original image
    and the decoded image"""

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
        elif fileExt in TiffPaths:
            # 16 bit tifffiles are not read correctly by Pillow
            image = tifffile.imread(str(ifile))
        elif fileExt == ".npz":
            image = np.load(ifile)
            image = image.f.arr_0  # Load data from inside file.
        elif fileExt in ImageArrayPaths:
            image = np.load(ifile)
        elif fileExt in niftiPaths:
            img = nib.load(ifile)
            image = img.get_data()
        else:
            return
        return image

    def latest_image_clipper(self, image, stats, factor):
        """
        This method clips the data max value and min value to 5 % of max value and min value of entire image distribution
        in stats file
        Parameters
        ----------
        image: image matrix
        stats: image statistics
        factor: interval
        Returns
        -------
        """
        extreme = max(abs(stats["max"]), abs(stats["min"]))
        max_value = factor * extreme
        min_value = -factor * extreme
        width, height = image.shape
        for x in range(width):
            for y in range(height):
                if image[x][y] > max_value:
                    image[x][y] = max_value
                elif image[x][y] < min_value:
                    image[x][y] = min_value

        return image

    def image_decoder(self, conf):
        idir = Path(conf.input_dir)
        odir = Path(conf.output_dir)
        stats = json.load(open(str(idir / "stats.json")))
        extreme = max(abs(stats["max"]), abs(stats["min"]))
        max_value = conf.factor * extreme
        min_value = -conf.factor * extreme
        lab = ColorMap(conf.lab_yaml, max_value, min_value)
        image_paths = idir.rglob("*.png")
        for i, image_path in enumerate(tqdm(image_paths)):
            image_name = os.path.splitext(image_path.name)[0]
            image_matrix = self.loader(image_path)
            width, height= image_matrix.shape
            final_matrix = np.zeros((width, height))
            for x in range(width):
                for y in range(height):
                    print(f"{image_matrix[x][y]}")
                    final_matrix[x][y] = lab.deconvert(image_matrix[x][y])
            #np.savez_compressed(str(odir/image_name), final_matrix)

            image = Image.fromarray(final_matrix)
            image.save(str(odir / image_name) + r".png")

        with open(str(odir/"stats.json"), "w") as sfile:
            json.dump(stats, sfile)

class Configurator:
    """this class is for configurating input and output"""
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--input_dir", default=os.path.dirname(os.path.abspath(__file__))+r"/input_images",
                                 help="use this command to pass the input directory where the images are present")
        self.parser.add_argument("--output_dir", default=os.path.dirname(os.path.abspath(__file__))+r"/output_images",
                                 help="use this command to pass the out put directory where you want images to be saved")
        self.parser.add_argument("--lab_yaml", default=os.path.dirname(os.path.abspath(__file__))+r"/lab.yaml",
                                 help="use this command to specify the location of lab.yaml")
        self.parser.add_argument("--factor", type=float, default=0.4,
                                 help="use this command to set the interval for clipping the image and colormap")

    def parse(self, args=None):
        """Parse the configuration"""
        self.conf = self.parser.parse_args(args=args)
        return self.conf

if __name__ == "__main__":

    idir = r"/home/venkat/"

    if (
        len(sys.argv) == 1
        or "--input_dir" not in str(sys.argv)
    ):
        sys.argv.append("-h")
    conf = Configurator().parse()
    colormap = Deconverter()
    colormap.image_decoder(conf)