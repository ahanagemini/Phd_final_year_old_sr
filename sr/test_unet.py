from pathlib import Path

from model import process

input_path_train = Path("/home/venkat/Documents/PiyushKumarProject/sr/tests/CutterOutput/train")
input_path_valid = Path("/home/venkat/Documents/PiyushKumarProject/sr/tests/CutterOutput/valid")
log_dir = Path("/home/venkat/Documents/PiyushKumarProject/sr/tests/Logger")

process(input_path_train, input_path_valid, log_dir)