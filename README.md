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

Running the trainer
========

# executing the code for training a network

The command for performing training:

```
python ./trainer.py --train=TRAIN_PATH --valid=VALID_PATH --log_dir=logs/ --architecture=edsr
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
