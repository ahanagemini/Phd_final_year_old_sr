"""
Usage:   downloader.py --download=all --output_directory=output_path
         downloader.py --help | -help | -h

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
from docopt import docopt
import sys
import shutil


def get_file_name(url):
    """

    :param url: file url
    :return: file_name, file_ext, odir
    """
    base_name = os.path.basename(url)
    base_name = str.split(base_name, sep="?")[0]
    file_name, file_ext = (
        str.split(base_name, sep=".")[0],
        ".".join(str.split(base_name, sep=".")[1:]),
    )

    return file_name, file_ext


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
    if file_ext == "tar" or file_ext == "tar.gz":
        tar = tarfile.open(file_tmp)
        tar.extractall(odir)



def bar_progress(current, total, width=80):
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()

def data_download(dataname, odir):
    """

    :param dataname: the name of the data to be downloaded or all
    :param odir: the path where the files will be extracted
    :return:
    """
    datadict = {
        "div2k": r"http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
        "earth1": r"https://www.dropbox.com/s/hzqcuct3phd7g5s/earth1.tar?dl=0",
        "earth2": r"https://www.dropbox.com/s/e9p0dt001mrej99/earth2.tar?dl=0",
        "BloodData": r"http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
        "chest_xray": r"https://www.dropbox.com/s/hzqcuct3phd7g5s/earth1.tar?dl=0",
        "Retinal_OCT": r"https://www.dropbox.com/s/e9p0dt001mrej99/earth2.tar?dl=0",
        "Medical_3D": r"https://www.dropbox.com/s/ri3cpsunqed32my/Medical_data.tar.gz?dl=0",
    }
    down_path = Path(os.getcwd()+r"/Download")
    if os.path.isdir(down_path):
        shutil.rmtree(down_path)
    os.mkdir(down_path)
    if dataname == "all":
        for i, url_key in enumerate(datadict):
            url = datadict[url_key]
            file_name, file_ext = get_file_name(url)
            file_down_path = down_path / (file_name+"."+file_ext)
            # file_tmp = wget.download(url, out=str(file_down_path), bar=bar_progress)
            os.system(f'wget {url} -O {file_down_path}')
            file_extraction(file_down_path, file_ext, odir)
            if url_key == "Medical_3D":
                files = os.listdir(odir / "Medical_3D")
                for compressed_file in files:
                    file_extraction(odir / "Medical_3D" / compressed_file, file_ext, odir)
                    os.remove(odir / "Medical_3D" / compressed_file)
    else:
        url = datadict[dataname]
        file_name, file_ext = get_file_name(url)
        file_down_path = down_path / (file_name+"."+file_ext)
        # file_tmp = wget.download(url, out=str(file_down_path), bar=bar_progress)
        os.system(f'wget {url} -O {file_down_path}')
        file_extraction(file_down_path, file_ext, odir)
    shutil.rmtree(down_path)

if __name__ == "__main__":
    arguments = docopt(__doc__)
    down = str(arguments["--download"])
    odir = Path(arguments["--output_directory"])
    data_download(down, odir)


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
