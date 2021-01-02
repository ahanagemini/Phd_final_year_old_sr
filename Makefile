PYTHON = python3

root_dir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
inputdir := "$(root_dir)/input_dir"
outputdir := "$(root_dir)/output_dir"
cuttingdir := "$(root_dir)/output_dir/cutting_out"
resumedir := "$(root_dir)/output_dir/edsr_16_64/edsr_16_64"
loggerdir := "$(root_dir)/output_dir/logger"
architecture := "edsr_16_64"

clean:
	echo "clean started"
	if [ -d inputdir ]; then rm -Rf inputdir; fi
	if [ -d outputdir ]; then rm -Rf outputdir; fi
	echo "clean ended"
	
run:
	mkdir -p $(outputdir);
	echo "download started"
	$(PYTHON) sr/downloader.py --download="earth1" --output_directory=$(inputdir)
	$(PYTHON) sr/downloader.py --download="slices" --output_directory=$(inputdir)
	echo "download ended"

	$(PYTHON) sr/zeroshotpreprocessing.py --input_dir_path=$(inputdir) --output_dir_path=$(outputdir) --cutting_output_dir_path=$(cuttingdir) --model_save=$(outputdir) --num_epochs=1 --log_dir=$(loggerdir) --architecture=$(architecture)
	echo "training ended"


resume:
	$(PYTHON) sr/zeroshotpreprocessing.py --input_dir_path=$(inputdir) --output_dir_path=$(outputdir) --cutting_output_dir_path=$(cuttingdir) --model_save=$(outputdir) --num_epochs=3 --log_dir=$(loggerdir) --architecture=$(architecture) --resume=$(resumedir)
	
		
setup:
	$(PYTHON) -m pip install -r requirements.txt
	
