import os
import sys
import json
import shutil
from pathlib import Path
import numpy as np
from  zeroshotpreprocessing import assert_stats, stat_calculator
from cutter import loader, matrix_cutter
import scipy.ndimage
from tqdm import tqdm
from configs import Config
from tester import evaluate

def process(conf):
    output_directory = Path(conf.cutting_output_dir_path)

    # deletting if cutting out existed already to avoid overlaps
    if os.path.isdir(output_directory):
        print("deleting existing path")
        shutil.rmtree(output_directory)
    input_directory = Path(conf.input_dir_path)
    stats = assert_stats(input_directory)

    parent_folders = os.scandir(input_directory)

    print("started image pair creation")
    for parent in parent_folders:
        parent = Path(parent)
        ldir_train = output_directory / "test" / "LR" / parent.name
        hdir_train = output_directory / "test" / "HR" / parent.name

        if not os.path.isdir(ldir_train):
            print("creating lr directory")
            os.makedirs(str(ldir_train))
        if not os.path.isdir(hdir_train):
            print("creating hr directory")
            os.makedirs(str(hdir_train))

        #save stat file
        with open(str(ldir_train / "stats.json"), "w") as sfile:
            json.dump(stats, sfile)

        with open(str(hdir_train / "stats.json"), "w") as sfile:
            json.dump(stats, sfile)
        input_files = list(parent.rglob("*.npz"))
        for i, image_path in enumerate(tqdm(input_files)):
            image_name = image_path.name
            lr_image_matrix = loader(image_path)
            hr_image_matrix = scipy.ndimage.zoom(lr_image_matrix, 4)
            np.savez_compressed(ldir_train / image_name, lr_image_matrix)
            np.savez_compressed(hdir_train / image_name, hr_image_matrix)

        print("image pair creation complete starting testing")
        test_path = output_directory / "test"
        best_model_save = Path(conf.model_save)
        best_model_save = best_model_save / conf.architecture
        best_model = sorted(list(best_model_save.rglob("*best_model.pt")))[-1]
        args = {
            "--input": test_path,
            "--output": conf.output_dir_path,
            "--architecture": conf.architecture,
            "--model": best_model,
            "--act": conf.act,
            "--lognorm": conf.lognorm,
            "--active": conf.active,
            "--save_slice": conf.save_slice,
            "--aspp": conf.aspp,
            "--dilation": conf.dilation,
            "kernel": True,
            "hr": True,
        }
        evaluate(args)

if __name__ == '__main__':
    if len(sys.argv) == 1 or "--input_dir_path" not in str(sys.argv) or "--output" not in str(sys.argv):
        sys.argv.append('-h')
    conf = Config().parse()
    process(conf)
