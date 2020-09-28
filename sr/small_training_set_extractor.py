"""Usage:   small_training_set_extractor.py --input_directory=input --output_directory=output
            small_training_set_extractor.py --help | -help | -h

    This file is for extracting a higher loos files calculated using L1 loss. The input folder structure for this file
     is the output of Cutter. Best to run this file after running Cutter
Arguments:
    --input_directory: Input directory containing images.
    --output_directory: The directory where the images are present after processing
Options:
    -h --help -h
"""

import json
import os
from pathlib import Path
import numpy as np
import scipy.ndimage
from tqdm import tqdm
from docopt import docopt

class TrainingSet_Extractor:
    def __init__(self, input_directory, output_directory, percentage=0.9):
        """
        This extractor extracts a small portion of dataset based on l1 error of images. The dataset should be a
        :param input_directory: Input path of the files
        :param output_directory: Output path for the small dataset
        :param percentage: How much percentage of dataset required. values between 0 and 1 where
               1 is 100% and 0 is 0%, default: 0.9 (90%)
        """
        self.input_directory = Path(input_directory)
        self.output_directory = Path(output_directory)
        self.percentage = percentage

    def l1_loss(self, matrix_1, matrix_2):
        """
        To calculate L1 loss between two matrices
        :param matrix_1: Image matrix
        :param matrix_2: Image matrix
        :return:
        """
        return np.mean(np.abs(matrix_1 - matrix_2))

    def loader(self, file_path):
        """

        :param file_path: the Image file path
        :return:
        """
        imatrix = np.load(file_path)
        imatrix = imatrix.f.arr_0
        return imatrix

    def loss_matrix_extract(self, hr_imatrix):
        """

        :param hr_matrix:
        :return: loss_val:
        """
        lr_imatrix = scipy.ndimage.zoom(scipy.ndimage.zoom(hr_imatrix, 0.25), 4.0)
        loss_val = self.l1_loss(lr_imatrix, hr_imatrix)
        return loss_val

    def loss_sorting_extracting(self, loss_list, percentage):
        loss_list = np.sort(loss_list)
        new_loss_list = loss_list[int(percentage * len(loss_list)) :]
        return new_loss_list

    def final_save_directory(self, loss_list, loss_key_map, key_matrix_map, stat_file):
        for loss_val in tqdm(loss_list, total=len(loss_list)):
            key = loss_key_map[loss_val]
            hr_imatrix, train_valid_test, distribution, file_name = key_matrix_map[key]
            opath = self.output_directory / train_valid_test / distribution
            if not os.path.isdir(opath):
                os.makedirs(opath)
            np.savez_compressed(opath / file_name, hr_imatrix)
            stats_path = opath / "stats.json"
            with open(stats_path, "w") as outfile:
                json.dump(stat_file, outfile)

    def extractor(self, input_list, stat_file):
        """

        :param input_list:
        :param stat_file:
        :return:
        """
        train_loss_list = []
        valid_loss_list = []
        test_loss_list = []
        key_matrix_map = {}
        key_train_loss_map = {}
        key_valid_loss_map = {}
        key_test_loss_map = {}
        key = 0
        for input_files in tqdm(input_list, total=len(input_list)):
            hr_imatrix = self.loader(file_path=input_files)
            folder_path = input_files.parent
            distribution = folder_path.name
            file_name = input_files.name
            train_valid_test = input_files.parent.parent.name.lower()
            if train_valid_test == "train":
                key_matrix_map[key] = (hr_imatrix, "train", distribution, file_name)
                loss_val = self.loss_matrix_extract(hr_imatrix)
                train_loss_list.append(loss_val)
                key_train_loss_map[loss_val] = key
            elif train_valid_test == "valid":
                key_matrix_map[key] = (hr_imatrix, "valid", distribution, file_name)
                loss_val = self.loss_matrix_extract(hr_imatrix)
                valid_loss_list.append(loss_val)
                key_valid_loss_map[loss_val] = key
            elif train_valid_test == "test":
                key_matrix_map[key] = (hr_imatrix, "test", distribution, file_name)
                loss_val = self.loss_matrix_extract(hr_imatrix)
                test_loss_list.append(loss_val)
                key_test_loss_map[loss_val] = key
            key = key + 1

        train_loss_list = self.loss_sorting_extracting(train_loss_list, self.percentage)
        valid_loss_list = self.loss_sorting_extracting(valid_loss_list, self.percentage)
        test_loss_list = self.loss_sorting_extracting(test_loss_list, self.percentage)

        self.final_save_directory(
            train_loss_list, key_train_loss_map, key_matrix_map, stat_file
        )
        self.final_save_directory(
            valid_loss_list, key_valid_loss_map, key_matrix_map, stat_file
        )
        self.final_save_directory(
            test_loss_list, key_test_loss_map, key_matrix_map, stat_file
        )

    def folder_extraction(self, input_directory):
        distribution_map = {}
        stat_distribution = {}
        distribution_list = {}
        input_files = list(input_directory.rglob("*.npz"))
        for ifile in tqdm(input_files, total=len(input_files)):
            folder = ifile.parent
            train_folder = folder.parent
            if folder.name not in distribution_map.keys():
                distribution_map[folder.name] = []
            if train_folder.name == "train":
                if folder.name not in distribution_list.keys():
                    distribution_list[folder.name] = folder
            distribution_map[folder.name].append(ifile)
        for i, fold in enumerate(distribution_list):
            folder = distribution_list[fold]
            with open(folder / "stats.json") as infile:
                stat_file = json.load(infile)
            stat_distribution[folder.name] = stat_file
        return distribution_map, stat_distribution

    def run(self):
        distribution_map, stat_distribution = self.folder_extraction(
            self.input_directory
        )
        for i, keys in enumerate(distribution_map):
            print("Processing {} distribution".format(keys))
            self.extractor(distribution_map[keys], stat_distribution[keys])


if __name__ == "__main__":
    arguments = docopt(__doc__)
    idir = Path(arguments["--input_directory"])
    odir = Path(arguments["--output_directory"])
    trainer = TrainingSet_Extractor(idir, odir, 0.9)
    trainer.run()
