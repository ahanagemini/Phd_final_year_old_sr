# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 22:41:51 2020

@author: sumanth
"""
from PIL import Image
import numpy as np
import os


def Loader(filePath):
    '''
    Parameters
    ----------
    filePath : image File path
        DESCRIPTION.

    Returns
    -------
    image : PillowImage
        DESCRIPTION.

    '''

    ImagePaths = [".jpg", ".png", ".jpeg", ".gif"]
    ImageArrayPaths = [".npy", ".npz"]
    fileExt = os.path.splitext(filePath)[1].lower()
    if fileExt in ImagePaths:
        image = Image.open(filePath)
    elif fileExt in ImageArrayPaths:
        image = Image.fromarray(np.uint8(np.load(filePath)))
    return image


def matrixcutter(imgPath, height=256, width=256):
    '''


    Parameters
    ----------
    imgPath : String
        image file.
    height : int
        height of the cut image.
    width : int
        width of the cut image.

    Returns
    -------
    images: List
    Contains list of pillow images cut in 256 width and 256 height.

    '''
    img = Loader(imgPath)
    images = []
    imgWidth, imgHeight = img.size
    for y in range(0, imgHeight, height):
        for x in range(0, imgWidth, width):
            xPrime = x + width
            yPrime = y + height
            box = (x, y, xPrime, yPrime)
            if xPrime > imgWidth:
                # exceededWidth is the difference between xPrime and original Image width
                exceededWidth = xPrime - imgWidth
                box = (x - exceededWidth, y, xPrime - exceededWidth, yPrime)
            if yPrime > imgHeight:
                # exceededHeight is the difference between yPrime and original Image height
                exceededHeight = yPrime - imgHeight
                box = (x, y - exceededHeight, xPrime, yPrime - exceededHeight)
            cutimg = img.crop(box)
            images.append(cutimg)
    return images











