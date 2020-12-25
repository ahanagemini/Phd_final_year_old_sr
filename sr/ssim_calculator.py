
import numpy as np
import os
import json
import argparse
import sys
import matplotlib.pyplot as plt
import pylab
from pathlib import Path
from skimage import metrics
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from mayavi import mlab

class SSIM_Calculator:
    """This class will calculate the loss between two images"""
    def loader(self, ifile):
        image = Image.open(ifile)
        image = image.convert("RGB")
        image = np.array(image)
        return image

    def plotter(self, hr_image, sr_image, odir):
        hr_plot_image = np.ravel(hr_image)
        sr_plot_image = np.ravel(sr_image)
        hr_x = [x for x in range(len(hr_plot_image))]
        sr_x = [x for x in range(len(sr_plot_image))]
        import matplotlib as mpl
        mpl.rcParams['agg.path.chunksize'] = 10000
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111)
        ax.plot(hr_x, hr_plot_image, label="hr")
        ax.plot(sr_x, sr_plot_image, label="sr")
        ax.legend()
        fig.savefig(odir+r"_lineplot.png")
        plt.close(fig=fig)

        diff_image = np.abs(hr_plot_image - sr_plot_image)
        diff_x = [x for x in range(len(diff_image))]
        import matplotlib as mpl
        mpl.rcParams['agg.path.chunksize'] = 10000
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(111)
        ax.scatter(diff_x, diff_image, label="diff_image")
        fig.savefig(odir +r"_diff_lineplot.png")
        plt.close(fig=fig)


    def writetext(self, img, image_output_file, l1, psnr, ssim):
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        font_path = os.path.abspath(os.path.expanduser("./font/dancing.ttf"))
        font = ImageFont.truetype(font_path, 15)
        draw.text((0, 0), "L1=" + str(l1), font=font, fill=(0, 0, 0))
        draw.text((0, 16), "PSNR=" + str(psnr), font=font, fill=(0, 0, 0))
        draw.text((0, 32), "SSIM=" + str(ssim), font=font, fill=(0, 0, 0))
        img.save(image_output_file)

    def ssim_calculate(self, conf):
        hr_dir, sr_dir, odir = Path(conf.hr_dir), Path(conf.sr_dir), Path(conf.output_dir)
        if not os.path.isdir(odir):
            os.makedirs(odir)
        image_paths = hr_dir.rglob("*.png")
        average_losses = {}
        ssim_list = []
        l1_loss_list = []
        psnr_loss_list = []
        for i, image_path in enumerate(tqdm(image_paths)):
            image_name = image_path.name
            sr_image_path = sr_dir / image_name

            hr_image = self.loader(image_path)
            sr_image = self.loader(sr_image_path)
            vmax = np.max(sr_image)
            with np.errstate(divide="ignore"):
                l1_loss = np.mean(np.abs(hr_image - sr_image))
                psnr_loss = metrics.peak_signal_noise_ratio(
                    hr_image, sr_image, data_range=vmax
                )
                ssim_loss = metrics.structural_similarity(
                    hr_image, sr_image, data_range=vmax,multichannel=True
                )
            out_sr = odir / image_name
            self.writetext(sr_image, out_sr, l1_loss, psnr_loss, ssim_loss)
            self.plotter(hr_image, sr_image, str(odir/(os.path.splitext(image_name)[0])))

            ssim_list.append(ssim_loss)
            psnr_loss_list.append(psnr_loss)
            l1_loss_list.append(l1_loss)

        average_losses["l1_loss"] = float(np.mean(l1_loss_list))
        average_losses["psnr"] = float(np.mean(psnr_loss_list))
        average_losses["ssim"] = float(np.mean(ssim_list))

        with open(str(odir/"average_losses.json"), "w") as sfile:
            json.dump(average_losses, sfile)

class Configurator:
    """this class is for configurating input and output"""
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--hr_dir", default=os.path.dirname(os.path.abspath(__file__))+r"/hr_images",
                                 help="use this command to pass the input directory where the hr images are present")
        self.parser.add_argument("--sr_dir", default = os.path.dirname(os.path.abspath(__file__))+r"/sr_images",
                                 help="use this command to pass the input directory where the sr images are present")
        self.parser.add_argument("--output_dir", default=os.path.dirname(os.path.abspath(__file__))+r"/output_images",
                                 help="use this command to pass the out put directory where you want images to be saved")

    def parse(self, args=None):
        """Parse the configuration"""
        self.conf = self.parser.parse_args(args=args)
        return self.conf

if __name__ == "__main__":

    if (
        len(sys.argv) == 1
        or "--hr_dir" and "--sr_dir" not in str(sys.argv)
    ):
        sys.argv.append("-h")
    conf = Configurator().parse()
    calculator = SSIM_Calculator()
    calculator.ssim_calculate(conf)