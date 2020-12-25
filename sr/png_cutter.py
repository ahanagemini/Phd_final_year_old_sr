import os
import sys
import json
import argparse
from PIL import Image
from pathlib import Path
import numpy as np
from tqdm import tqdm

class PNG_Cutter:
    """This class main purpose is to load png images and generate pieces of image of specified cut size"""

    def png_matrix_cutter(self, img, width=256, height=256):
        """


        Parameters
        ----------
        image : TYPE
            DESCRIPTION.
        height : TYPE, optional
            DESCRIPTION. The default is 256.
        width : TYPE, optional
            DESCRIPTION. The default is 256.

        Returns
        -------
        None.

        """
        images = []
        img_height, img_width, img_channels = img.shape

        # check if images have 256 width and 256 height if it does skip cutting
        if img_height <= height and img_width <= width:
            return [(0, 0, img)]

        for i, ih in enumerate(range(0, img_height, height)):
            for j, iw in enumerate(range(0, img_width, width)):
                posx = iw
                posy = ih
                if posx + width > img_width:
                    posx = img_width - width
                if posy + height > img_height:
                    posy = img_height - height

                cutimg = img[posy: posy + height, posx: posx + width, :]
                cutimg_height, cutimg_width, cutimg_channels = cutimg.shape
                assert cutimg_height == height and cutimg_width == width
                images.append((i, j, cutimg))
        return images

    def loader(self, ifile):
        """
        This loader will accept only png files

        Parameters
        ----------
        file : TYPE
            DESCRIPTION.

        Returns
        -------
        imageArray : TYPE
            DESCRIPTION.

        """

        image = Image.open(ifile)
        image = np.array(image)
        return image

    def png_extractor(self, conf):
        idir = Path(conf.input_dir)
        odir = Path(conf.output_dir)
        if not os.path.isdir(odir):
            os.makedirs(odir)
        image_paths = idir.rglob("*.png")
        total_sum = 0.0
        total_square_sum = 0.0
        total_count = 0.0
        image_min = 0
        image_max = 0
        for i, image_path in enumerate(tqdm(image_paths)):
            image_name = os.path.splitext(image_path.name)[0]
            image = self.loader(image_path)
            matrix_vector = np.asarray(image).reshape(-1)
            square_vector = np.square(matrix_vector, dtype=np.float)
            matrix_sum = np.sum(matrix_vector, dtype=np.float)
            square_sum = np.sum(square_vector, dtype=np.float)
            matrix_count = len(matrix_vector)

            # this information is for total mean calculation
            total_sum = total_sum + matrix_sum
            total_square_sum = total_square_sum + square_sum
            total_count = total_count + matrix_count

            # maximum and minimum
            matrix_max = np.max(matrix_vector)
            matrix_min = np.min(matrix_vector)
            if image_max < matrix_max:
                image_max = matrix_max

            if image_min > matrix_min:
                image_min = matrix_min

            mlist = self.png_matrix_cutter(image, width=conf.width, height=conf.height)
            for i, j, mat in mlist:
                fname = str(image_name) + "_" + str(i) + "_" + str(j)
                mat = Image.fromarray(mat)
                mat.save(str(odir / (fname+r".png")))

        # stats
        total_mean = total_sum / total_count
        total_variance = (total_square_sum / total_count) - (total_sum / total_count) ** 2
        stats = {}
        stats["mean"] = total_mean
        stats["std"] = np.sqrt(total_variance)
        stats["max"] = float(image_max)
        stats["min"] = float(image_min)
        with open(str(odir / "stats.json"), "w") as sfile:
            json.dump(stats, sfile)

class Configurator:
    """This is the configurator class"""
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--input_dir", default=os.path.dirname(os.path.abspath(__file__)+r"/input_directory"),
                                 help="This command is used to set input_dir")
        self.parser.add_argument("--output_dir", default=os.path.dirname(os.path.abspath(__file__))+r"/output_directory",
                                 help="This command is used to set output_dir")
        self.parser.add_argument("--height", type=int, default=256,
                                 help="Use this command to specify height of cut image")
        self.parser.add_argument("--width", type=int, default=256,
                                 help="Use this command to specify width of cut image")

    def parse(self, args=None):
        """Parse the configuration"""
        self.conf = self.parser.parse_args(args=args)
        return self.conf

if __name__=="__main__":
    if (
        len(sys.argv) == 1
        or "--input_dir" not in str(sys.argv)
    ):
        sys.argv.append("-h")
    conf = Configurator().parse()
    colormap = PNG_Cutter()
    colormap.png_extractor(conf)

