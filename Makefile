PYTHON = python3

epochs := 350
architecture := "vgg"
root_dir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
inputdir := "$(root_dir)/input_dir"
outputdir := "$(root_dir)/output_dir"
cuttingdir := "$(root_dir)/output_dir/cutting_out"
resumedir := "$(root_dir)/output_dir/$(architecture)/$(architecture)"
loggerdir := "$(root_dir)/output_dir/logger"
download_dir := "$(root_dir)/Download"
pretrained_model := "$(root_dir)/input_dir/edsr_64_64"


clean:
	echo "clean started"
	if [ -d $(inputdir) ]; then rm -Rf $(inputdir); fi
	if [ -d $(outputdir) ]; then rm -Rf $(outputdir); fi
	if [ -d $(download_dir) ]; then rm -Rf $(download_dir); fi
	echo "clean ended"

run:
	echo "download started"
	$(PYTHON) sr/downloader.py --download="earth1" --output_directory=$(inputdir)
	$(PYTHON) sr/downloader.py --download="slices" --output_directory=$(inputdir)
	$(PYTHON) sr/downloader.py --download="edsr_64" --output_directory=$(inputdir)
	echo "download ended"

	$(PYTHON) sr/zeroshotpreprocessing.py --input_dir_path=$(inputdir) --output_dir_path=$(outputdir) --cutting_output_dir_path=$(cuttingdir) --model_save=$(outputdir) --num_epochs=$(epochs) --log_dir=$(loggerdir) --architecture=$(architecture) --pretrained_model_path=$(pretrained_model) --vgg=True
	echo "training ended"


resume:
	$(PYTHON) sr/zeroshotpreprocessing.py --input_dir_path=$(inputdir) --output_dir_path=$(outputdir) --cutting_output_dir_path=$(cuttingdir) --model_save=$(outputdir) --num_epochs=$(epochs) --log_dir=$(loggerdir) --architecture=$(architecture) --resume=$(resumedir) --pretrained_model=$(pretrained_model) --vgg=True
	
		
setup:
	$(PYTHON) -m pip install -r requirements.txt

test:

	
