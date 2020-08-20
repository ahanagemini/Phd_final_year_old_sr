# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 22:41:51 2020

@author: sumu1
"""
from PIL import Image
import numpy as np


def Loader(file):
    '''


    Parameters
    ----------
    file : TYPE
        DESCRIPTION.

    Returns
    -------
    imageArray : TYPE
        DESCRIPTION.

    '''

    ImagePaths = [".jpg", ".png", ".jpeg", ".gif"]
    ImageArrayPaths = [".npy", ".npz"]
    fileExt = os.path.splitext(file.name)[1].lower()
    if fileExt in ImagePaths:
        image = Image.open(file)
    elif fileExt in ImageArrayPaths:
        image = Image.fromarray(np.uint8(np.load(file)))
    return image


def imageCutter(img, width=256, height=256):
    '''


    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    height : TYPE, optional
        DESCRIPTION. The default is 256.
    width : TYPE, optional
        DESCRIPTION. The default is 256.

    Returns
    -------
    None.

    '''
    images = []
    imgWidth, imgHeight = img.size
    for ih in range(0, imgHeight, height):
        for iw in range(0, imgWidth, width):
            box = (iw, ih, imageCutWidth, imageCutHeight)

            '''
            if iw + width > imgwidth : 
                boxL = (imgwidth - width, imgwidth)
            if ih + height > imgheight : 
                box = (box[0], box[1], imgheight-height, imgheight)
            if iw + width > imgwidth : 
                boxL = (imgwidth - width, imgwidth, box[1], box[2])
            if ih + height > imgheight : 
                box = (box[0], imgheight-height, box[2], imgheight)
            '''
            cutimg = img.crop(box)
            cutimgWidth, cutimgHeight = cutimg.size
            images.append(cutimg)
    return images


def matrixcutter(imgPath):
    '''


    Parameters
    ----------
    imgPath : TYPE
        DESCRIPTION.
    height : TYPE
        DESCRIPTION.
    width : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    for file in os.scandir(path):
        if file.is_dir():
            subFolders.append(file.path)
        if file.is_file():
            img = Loader(file)
            images = imageCutter(img, 256, 256)
    for dir in list(subFolders):
        subF = matrixcutter(dir)
        subFolders.extend(subF)
    return subFolders










