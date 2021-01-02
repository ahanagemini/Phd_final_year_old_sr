PYTHON = python3

root_dir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
inputdir := "$(root_dir)/sr/input_dir"
outputdir := "$(root_dir)/sr/output_dir"
cuttingdir := "$(outputdir)/cutting_out"
resumedir := "$(outputdir)/saved_models/edsr_16_64/edsr_16_64"
loggerdir := "$(outputdir)/logger"

clean:
	echo "clean started"
	if [ -d inputdir ]; then rm -Rf inputdir; fi
	if [ -d outputdir ]; then rm -Rf outputdir; fi
	echo "clean ended"
	
run:
	echo "download started"
	$(PYTHON) sr/downloader.py --download="earth1" --output_directory=inputdir
	echo "download ended"
	
	echo "training started"
	$(PYTHON) sr/zeroshotpreprocessing.py --input_dir_path=inputdir --output_dir_path=outputdir --cutting_output_dir_path=cuttingdir --model_save=outputdir --num_epochs=5 --log_dir=loggerdir --architecture="edsr_16_64"
	echo "training ended"
	
setup:
	$(PYTHON) -m pip install -r requirements.txt
	
