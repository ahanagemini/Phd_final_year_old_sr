
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

    def writetext_for_stack(self, img_file):
        img = Image.open(img_file)
        width, height = img.size
        draw = ImageDraw.Draw(img)
        font_path = os.path.abspath(os.path.expanduser('./font/dancing.ttf'))
        font = ImageFont.truetype(font_path, 15)
        draw.text((0, 0), "SR", font=font)
        draw.text((width / 2, 0), "HR", font=font, fill=(0, 0, 255))
        draw.text((0, width / 2), "Error", font=font, fill=(0, 0, 255))
        draw.text((height / 2, width / 2), "LR", font=font, fill=(0, 0, 255))
        img.save(img_file)


    def check_dif_matrix(self, hr_image, sr_image):
        width, height, channels = hr_image.shape
        diff_matrix_1 = np.zeros((width, height, channels), dtype=np.uint8)
        for x in range(width):
            for y in range(height):
                for z in range(channels):
                    diff_matrix_1[x][y][z] = np.abs(hr_image[x][y][z] - sr_image[x][y][z])

        diff_matrix_2 = np.abs(hr_image - sr_image)

        for x in range(width):
            for y in range(height):
                for z in range(channels):
                    assert diff_matrix_1[x][y][z] == diff_matrix_2[x][y][z]
    
    def error_image_plot(self, hr_image, sr_image, odir):
        diff_image = np.abs(hr_image - sr_image)
        diff_image = Image.fromarray(diff_image)
        lr_image = Image.fromarray(hr_image)
        lr_image = lr_image.resize((64, 64), resample=Image.BICUBIC)
        lr_image = lr_image.resize((256, 256), resample=Image.BICUBIC)

        horizontal_stack_1 = np.hstack([sr_image, hr_image])
        horizontal_stack_2 = np.hstack([diff_image, lr_image])
        stacked_error = np.vstack([horizontal_stack_1, horizontal_stack_2])
        odir = odir + r"_stacked.png"
        plt.imsave(odir, stacked_error, vmax=np.max(sr_image), vmin=np.min(sr_image))
        self.writetext_for_stack(odir)

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
        diff_image = sorted(diff_image)
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
        top_right_pixel = {}
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
                    hr_image, sr_image, data_range=vmax, multichannel=True
                )
            out_sr = odir / image_name
            self.check_dif_matrix(hr_image, sr_image)
            diff_image = np.abs(np.subtract(hr_image, sr_image))
            self.writetext(sr_image, out_sr, l1_loss, psnr_loss, ssim_loss)
            #self.plotter(hr_image, sr_image, str(odir/(os.path.splitext(image_name)[0])))
            self.error_image_plot(hr_image, sr_image, str(odir/(os.path.splitext(image_name)[0])))
            top_right_pixel[image_name+"_sr"] = sr_image[0][-1][:].tolist()
            top_right_pixel[image_name+"_err"] = diff_image[0][-1][:].tolist()
            top_right_pixel[image_name+"_hr"] = hr_image[0][-1][:].tolist()
            ssim_list.append(ssim_loss)
            psnr_loss_list.append(psnr_loss)
            l1_loss_list.append(l1_loss)

        average_losses["l1_loss"] = float(np.mean(l1_loss_list))
        average_losses["psnr"] = float(np.mean(psnr_loss_list))
        average_losses["ssim"] = float(np.mean(ssim_list))

        with open(str(odir/"average_losses.json"), "w") as sfile:
            json.dump(average_losses, sfile)
        with open(str(odir/ "top_right_pixel.json"), "w") as sfile:
            json.dump(top_right_pixel, sfile)

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