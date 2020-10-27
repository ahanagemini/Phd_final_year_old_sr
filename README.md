# sr
Super Resolution


cutter
======

Input for cutter is a directory which has subdirectories, perhaps for different daat domains. Lets call the input directory, idir. Then:

idir has subdirectories div2k, earth1, earth2, MPRAGE (for medical), b0_MUSE

such that each of these subdirectories has matrices of different sizes.

testdata
========

# Data directory structure.

A test directory consists of the following structure:

data/ directory must contain 3 subdirectories: test, train and validate.

Each of these directories must have subdirectories for each type of data. For example patient1, patient2 etc.
These directories must be disjoint for test/train/validate. So if patient1 is in test, it cant be in train or validate.

In these directories there must exist a .json file and multiple .npz files containing 256 x 256 x float data.

Running the zeroshotpreprocessing
=======

# data directory structure

A directory with image npz files. the output of this folder will be OutputDirectory-> Train/Valid -> LR and HR -> image
npz files. In both lr and hr there will be a stats.json file but in lr there will be a kernel file as well. The kernel
file in training will be used to downscale a image based on scale factor. The default scale factor is 4x. 

# Usage:
'''
python3 zeroshotpreprocessing.py --input_dir_path="/home/venkat/Documents/PiyushKumarProject/Data/newsrtest/slices" --output_dir_path="/home/venkat/Documents/PiyushKumarProject/KernelResult" --n_resize=10 --kernel_factor='--X4' --num_epochs=100 --architecture="edsr_16_64"

'''

Running the trainer
========

# executing the code for training a network

The command for performing training:

```
 python trainer.py --train="/home/venkat/Documents/PiyushKumarProject/KernelResult/train" --valid="/home/venkat/Documents/PiyushKumarProject/KernelResult/valid" --log_dir="/home/venkat//Documents/PiyushKumarProject/Logger" --num_epochs=10 --architecture="edsr_8_256" --act="leakyrelu" --kernel_factor="--X4"

```
The TRAIN_PATH and VALID_PATH contain *.npz files and a stats.json file
Possible architectures are edsr, unet and axial

Running the tester
========

# executing the code for testing a network

The command for performing testing:

```
./sr/tester hr --input=TEST_PATH --output=OUTPUT_PATH --model=MODEL_PATH --architecture=edsr
```
The TEST contains *.npz files and a stats.json file
Possible architectures are edsr, unet and axial

tensorboard
========

# running tensor board.

install tensorboard using 
pip install tensorflow

run tensorboard using 
tensorboard --logdir=path
where path is the place where tensorboard will recursively look for .tfevents file

Go to the URL it provides OR to http://localhost:6006/ to look for output.
