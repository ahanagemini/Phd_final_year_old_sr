# sr
Super Resolution


cutter
======

testdata
========

# Data directory structure.

A test directory consists of the following structure:

data/ directory must contain 3 subdirectories: test, train and validate.

Each of these directories must have subdirectories for each type of data. For example patient1, patient2 etc.
These directories must be disjoint for test/train/validate. So if patient1 is in test, it cant be in train or validate.
