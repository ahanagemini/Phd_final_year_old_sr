# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 22:41:51 2020

@author: sumu1
"""
from PIL import Image
import numpy as np
    
def Loader(self, file):
    ImagePaths = [".jpg", ".png", ".jpeg", ".gif"]
    ImageArrayPaths = [".npy", ".npz"]
    fileExt = os.path.splitext(file.name)[1].lower()
    if fileExt in ImagePaths:
        imageArray = np.array(Image.open(file)
    elif fileExt in ImageArrayPaths:
        imageArray = np.load(file)
    
def matrixcutter(self, imgPath, height, width):
    for file in os.scandir(imgpath):
        img = self.Loader(file)
        images=[]
        imgWidth, imgHeight = img.size
        for ih in range(0, imgHeight, height):
            for iw in range(0, imgWidth, width):
                box = (iw, ih, imageCutWidth, imageCutHeight)
                
                if iw + width > imgwidth : 
                    boxL = (imgwidth - width, imgwidth)
                if ih + height > imgheight : 
                    box = (box[0], box[1], imgheight-height, imgheight)
                if iw + width > imgwidth : 
                    boxL = (imgwidth - width, imgwidth, box[1], box[2])
                if ih + height > imgheight : 
                    box = (box[0], imgheight-height, box[2], imgheight)
                
                cutimg = img.crop(box)
                cutimgWidth, cutimgHeight = cutimg.size 
                images.append(cutimg)
        
            
            
         
        

