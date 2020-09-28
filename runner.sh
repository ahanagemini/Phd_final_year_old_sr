train_path=/home/ahana/testCutterOutput/train/DIV2K_train_HR
valid_path=/home/ahana/testCutterOutput/valid/DIV2K_train_HR
wget https://www.dropbox.com/s/wp3s9xlhm2t52qi/testCutterOutput.tar.gz?dl=1 -O testCutterOutput.tar.gz
tar -xvf testCutterOutput.tar.gz
cd sr/sr
python ./trainer.py --train=$train_path --valid=$valid_path --log_dir=logs/ --architecture=unet

