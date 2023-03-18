from skimage import data, io, filters
from skimage.color import rgb2gray
from skimage import exposure

from PySide6.QtGui import QImage, qRgb
import numpy as np
import logging
import VAImage
import cv2 as cv


def filterAll(func,imageDictionary):
    # filteredImageList = {}
    # if bool(imageDictionary):
    #     fileName=list(imageDictionary.keys())[0]
    #     filePath=list(imageDictionary.values())[0]
    #     folterPath=filePath.replace(fileName,"")
    #     folterPath=VAImage.createFolder(folterPath)
    # else:
    #     return filteredImageList
    for imageName, imagePath in imageDictionary.items():
        image = func(imagePath)
        imagePath=VAImage.saveImage(image, folterPath, imageName )
        # filteredImageList[imageName] = imagePath
    # return filteredImageList
def filterAll(func,imageDictionary,*params):
    #filteredImageList = {}
    #if bool(imageDictionary):
    #    fileName=list(imageDictionary.keys())[0]
    #    filePath=list(imageDictionary.values())[0]
    #     folterPath = filePath.replace(fileName,"")
    #     folterPath=VAImage.createFolder(folterPath)
    # else:
    #     return filteredImageList
    for imageName, imagePath in imageDictionary.items():
        image = func(imagePath,*params)
        imagePath=VAImage.saveImage(image, folterPath, imageName)
        #filteredImageList[imageName] = imagePath
    #return filteredImageList

def rgbToGray(imgPath):
    img = cv.imread(imgPath)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #grayImg = np.uint8(rgb2gray(img) * 255)
    return img

def histogramGlobalEqualize(imgPath):
    img = cv.imread(imgPath)
    R, G, B = cv.split(img)
    output1_R = cv.equalizeHist(R)
    output1_G = cv.equalizeHist(G)
    output1_B = cv.equalizeHist(B)
    img = cv.merge((output1_R, output1_G, output1_B))
    return img


def histogramLocalEqualize(imgPath):
    img = cv.imread(imgPath)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    R, G, B = cv.split(img)
    output1_R = clahe.apply(R)
    output1_G = clahe.apply(G)
    output1_B = clahe.apply(B)
    img = cv.merge((output1_R, output1_G, output1_B))
    return img

def otsuBinarization(imgPath,*params):
    img = cv.imread(imgPath)
    R, G, B = cv.split(img)
    size=params[0]
    blur_R = cv.GaussianBlur(R, (size, size), 0)
    ret3_R, th_R = cv.threshold(blur_R, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    blur_G = cv.GaussianBlur(G, (size, size), 0)
    ret3_G, th_G = cv.threshold(blur_G, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    blur_B = cv.GaussianBlur(B, (size, size), 0)
    ret3_B, th_B = cv.threshold(blur_B, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    img = cv.merge((th_R, th_G, th_B))
    return img

def adaptiveGaussianTreshold(imgPath,*params):
    img = cv.imread(imgPath)
    R, G, B = cv.split(img)
    size=params[0]
    blur_R = cv.medianBlur(R, size)
    output1_R = cv.adaptiveThreshold(blur_R, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.ADAPTIVE_THRESH_GAUSSIAN_C, size, 2)
    blur_G = cv.medianBlur(G, size)
    output1_G = cv.adaptiveThreshold(blur_G, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.ADAPTIVE_THRESH_GAUSSIAN_C,size, 2)
    blur_B = cv.medianBlur(B, size)
    output1_B = cv.adaptiveThreshold(blur_B, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.ADAPTIVE_THRESH_GAUSSIAN_C, size, 2)
    img = cv.merge((output1_R, output1_G, output1_B))
    return img
def adaptiveMedianFilter(imgPath,*params):
    size = params[0]
    distance = params[1]
    img = cv.imread(imgPath)
    img_R, img_G, img_B = cv.split(img)

    blur_R = cv.GaussianBlur(img_R, (size, size), 0)
    blur_G = cv.GaussianBlur(img_G, (size, size), 0)
    blur_B = cv.GaussianBlur(img_B, (size, size), 0)

    half_distance = distance / 2
    distance_d = distance
    (iH, iW) = img.shape[:2]
    pad = int((size - 1) / 2)
    imgb_R = cv.copyMakeBorder(img_R, pad, pad, pad, pad, cv.BORDER_REPLICATE)
    imgb_G = cv.copyMakeBorder(img_G, pad, pad, pad, pad, cv.BORDER_REPLICATE)
    imgb_B = cv.copyMakeBorder(img_B, pad, pad, pad, pad, cv.BORDER_REPLICATE)
    output_R = np.zeros((iH, iW), dtype="uint8")
    output_G = np.zeros((iH, iW), dtype="uint8")
    output_B = np.zeros((iH, iW), dtype="uint8")

    for y in range(iH):
        for x in range(iW):
            return

#suace filter
def speededUpAdaptiveContrastEnhancement(imgPath,*params):
    size = params[0]
    distance = params[1]
    img = cv.imread(imgPath)
    img_R, img_G, img_B = cv.split(img)

    blur_R = cv.GaussianBlur(img_R, (size, size), 3)
    blur_G = cv.GaussianBlur(img_G, (size, size), 3)
    blur_B = cv.GaussianBlur(img_B, (size, size), 3)

    half_distance = distance / 2
    distance_d = distance
    (iH, iW) = img.shape[:2]
    pad = int((size - 1) / 2)
    imgb_R = cv.copyMakeBorder(blur_R, pad, pad, pad, pad, cv.BORDER_REPLICATE)
    imgb_G = cv.copyMakeBorder(blur_G, pad, pad, pad, pad, cv.BORDER_REPLICATE)
    imgb_G = blur_G
    imgb_B = cv.copyMakeBorder(blur_B, pad, pad, pad, pad, cv.BORDER_REPLICATE)
    output_R = np.zeros((iH, iW), dtype="uint8")
    output_G = np.zeros((iH, iW), dtype="uint8")
    output_B = np.zeros((iH, iW), dtype="uint8")
    for y in range(iH):
        for x in range(iW):
            pix_G = img_G[y, x]
            adjusterPix_G = imgb_G[y, x]
            if (pix_G-adjusterPix_G)>distance_d:
                adjusterPix_G = adjusterPix_G + (pix_G-adjusterPix_G)*0.5
            b_G=adjusterPix_G+half_distance
            b_G= b_G > 255 and 255 or b_G
            a_G= b_G - distance
            a_G = a_G < 0 and 0 or a_G
            if (pix_G >= a_G) and (pix_G <= b_G):
                output_G[y, x] = ((pix_G-a_G)/distance_d)*255
            elif pix_G < a_G:
                output_G[y, x] = 0
            elif pix_G > b_G:
                output_G[y, x] = 255
    img = cv.merge((output_R, output_G, output_B))
    return img




