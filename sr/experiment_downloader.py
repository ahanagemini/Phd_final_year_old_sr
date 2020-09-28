"""
Usage:   experiment_downloader.py --download=all --output_directory=output_path
         experiment_downloader.py --help | -help | -h

    This script is for downloading and extracting files
Arguments:
    --download: download files or all.
    --output_directory: the path where the extracted files will be present
Options:
    -h --help -h
"""
import os
import zipfile
import tarfile
from pathlib import Path
import wget
from wget import bar_thermometer
from docopt import docopt


def get_file_name(url, odir):
    """

    :param url: file url
    :param odir: output directory
    :return: file_name, file_ext, odir
    """
    base_name = os.path.basename(url)
    base_name = str.split(base_name, sep="?")[0]
    file_name, file_ext = (
        str.split(base_name, sep=".")[0],
        str.split(base_name, sep=".")[1],
    )
    odir = odir / file_name

    return file_name, file_ext, odir


def file_extraction(file_tmp, file_ext, odir):
    """

    :param file_tmp: the downloaded file
    :param file_ext: the extension of the file
    :param odir: the output directory of the file
    :return:
    """
    if not os.path.isdir(odir):
        os.makedirs(odir)
    if file_ext == "zip":
        with zipfile.ZipFile(file_tmp) as zip_ref:
            zip_ref.extractall(odir)
    if file_ext == "tar":
        tar = tarfile.open(file_tmp)
        tar.extractall(odir)


def data_download(dataname, odir):
    """

    :param dataname: the name of the data to be downloaded or all
    :param odir: the path where the files will be extracted
    :return:
    """
    datadict = {
        "div2k": r"http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
        "medical": r"https://www.dropbox.com/s/ri3cpsunqed32my/Medical_data.tar.gz?dl=0",
    }
    if dataname == "all":
        for i, url_key in enumerate(datadict):
            url = datadict[url_key]
            file_name, file_ext, odir = get_file_name(url, odir)
            file_tmp = wget.download(url, bar=bar_thermometer)
            file_extraction(file_tmp, file_ext, odir)
    else:
        url = datadict[dataname]
        file_name, file_ext, odir = get_file_name(url, odir)
        file_tmp = wget.download(url)
        file_extraction(file_tmp, file_ext, odir)


if __name__ == "__main__":
    arguments = docopt(__doc__)
    down = str(arguments["--download"])
    odir = Path(arguments["--output_directory"])
    downloader = data_download(down, odir)


"""
print("Enter the download links and press ctrl + d to stop reading\n")
url_list = sys.stdin.readlines()
output_directory = Path(os.getcwd() + r"/Image_Input_Directory")

if not os.path.isdir(output_directory):
    os.makedirs(output_directory)

download_directory = Path(os.getcwd()+"/download")
if not os.path.isdir(download_directory):
    os.makedirs(download_directory)

for url in url_list:
    base_name = os.path.basename(url)
    base_name = str.split(base_name, sep='?')[0]
    file_name, file_ext = str.split(base_name, sep=".")[0], str.split(base_name, sep=".")[1]
    file_down = str(download_directory / file_name)
    out_path = output_directory / file_name
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    print("downloading file from first url {}".format(url))
    file_tmp = urllib.urlretrieve(url, "download.zip")

    print("Extracting zip_file")
    with zipfile.ZipFile("download.zip", "r") as Z:
        Z.extractall(out_path, )
"""
