# sr
Super Resolution


cutter
======

Input for cutter is a directory which has subdirectories, perhaps for different patients. Lets call the input directory, idir. Then:

idir has subdirectories patient1, patient2, ... , patientn

such that each of these subdirectories has matrices of different sizes.

testdata
========

# Data directory structure.

A test directory consists of the following structure:

data/ directory must contain 3 subdirectories: test, train and validate.

Each of these directories must have subdirectories for each type of data. For example patient1, patient2 etc.
These directories must be disjoint for test/train/validate. So if patient1 is in test, it cant be in train or validate.

In these directories there must exist a .json file and multiple .npz files containing 256 x 256 x float data.
