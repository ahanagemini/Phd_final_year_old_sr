"""Usage:   small_training_set_extractor.py --input_directory=input --output_directory=output
                                            --train_size=2700 --valid_size=150 --test_size=150
            small_training_set_extractor.py --help | -help | -h

    This file is for extracting a higher loos files calculated using L1 loss. The input folder structure for this file
     is the output of Cutter. Best to run this file after running Cutter
Arguments:
    --input_directory: Input directory containing images.
    --output_directory: The directory where the images are present after processing
    --percentage:       Percentage of data to be extracted
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
    def __init__(self, input_directory, output_directory, train_size, valid_size, test_size):
        """
        This extractor extracts a small portion of dataset based on l1 error of images. The dataset should be a
        :param input_directory: Input path of the files
        :param output_directory: Output path for the small dataset
        :param train_size: number of training sample sto be extracted
        :param valid_size: number of validation samples to be extracted
        :param test_size: number of test samples to be extracted
        """
        self.input_directory = Path(input_directory)
        self.output_directory = Path(output_directory)
        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size

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

    def loss_sorting_extracting(self, key_loss_map, size):
        sorted_keys = sorted(key_loss_map, key=key_loss_map.__getitem__, reverse=True)
        new_key_list = sorted_keys[:size]
        new_key_loss_map = {k: key_loss_map[k] for k in new_key_list}
        return new_key_loss_map

    def final_save_directory(self, loss_key_map, key_matrix_map, stat_file):
        for key in tqdm(loss_key_map, total=len(loss_key_map.keys())):
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
                key_train_loss_map[key] = loss_val
            elif train_valid_test == "valid":
                key_matrix_map[key] = (hr_imatrix, "valid", distribution, file_name)
                loss_val = self.loss_matrix_extract(hr_imatrix)
                valid_loss_list.append(loss_val)
                key_valid_loss_map[key] = loss_val
            elif train_valid_test == "test":
                key_matrix_map[key] = (hr_imatrix, "test", distribution, file_name)
                loss_val = self.loss_matrix_extract(hr_imatrix)
                test_loss_list.append(loss_val)
                key_test_loss_map[key] = loss_val
            key = key + 1

        key_train_loss_map = self.loss_sorting_extracting(key_train_loss_map, self.train_size)
        key_valid_loss_map = self.loss_sorting_extracting(key_valid_loss_map, self.valid_size)
        key_test_loss_map = self.loss_sorting_extracting(key_test_loss_map, self.test_size)

        self.final_save_directory(
            key_train_loss_map, key_matrix_map, stat_file
        )
        self.final_save_directory(
            key_valid_loss_map, key_matrix_map, stat_file
        )
        self.final_save_directory(
            key_test_loss_map, key_matrix_map, stat_file
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
    train_size = int(arguments["--train_size"])
    valid_size = int(arguments["--valid_size"])
    test_size = int(arguments["--test_size"])
    trainer = TrainingSet_Extractor(idir, odir, train_size, valid_size, test_size)
    trainer.run()
