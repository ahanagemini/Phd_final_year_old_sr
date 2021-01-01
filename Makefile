PYTHON = python3

root_dir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
inputdir := "$(root_dir)/sr/input_dir"
outputdir := "$(root_dir)/sr/output_dir"


run:
	
	
	@echo "download"
	wget -P inputdir "https://www.dropbox.com/s/hzqcuct3phd7g5s/earth1.tar?dl=0" | find inputdir -name '*.tar*' -execdir tar -C inputdir -xvf '{}'\;
	
	
