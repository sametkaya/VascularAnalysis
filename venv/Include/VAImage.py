from PySide2.QtGui import QImage,qRgb
import numpy as np
from skimage import io
import os
import cv2 as cv
def toQImage(im, copy=False):
    gray_color_table = [qRgb(i, i, i) for i in range(256)]
    if im is None:
        return QImage()
    if im.dtype == np.uint8:
        if len(im.shape) == 2:
            qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_Indexed8)
            qim.setColorTable(gray_color_table)
            return qim.copy() if copy else qim

        elif len(im.shape) == 3:
            if im.shape[2] == 3:
                qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGB888);
                return qim.copy() if copy else qim
            elif im.shape[2] == 4:
                qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_ARGB32);
                return qim.copy() if copy else qim
def saveImage(im,folterPath,imageName):
    imagePath = os.path.join(folterPath, imageName)
    io.imsave(imagePath,im)
    return imagePath
def copyImage(sourceImagePath,destinationImagPath):
    img=cv.imread(sourceImagePath)
    io.imsave(destinationImagPath,img)
    return destinationImagPath
def createFolder(folderPath,additionFolter):
    folderPath=removeLastSlash(folderPath)
    if not folderPath.endswith(additionFolter):
        folderPath = os.path.join(folderPath, additionFolter)
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
    return folderPath
def findCountOfDirectoryStartWithName(folderPath, name):
    if not os.path.exists(folderPath):
        return -1
    count =0
    for path in os.listdir(folderPath):
        if os.path.isdir(path):
            fName=os.path.basename(os.path.dirname(path))
            if path.startswith(name):
                count+=1

    return count


    return len([name for name in os.listdir(folderPath) if os.path.isdir(os.path.join(folderPath, folderName))])

def deleteAllInFolder(folderPath):
    if not os.path.exists(folderPath):
        return
    for fileObject in os.listdir(folderPath):
        fileObjectPath = os.path.join(folderPath, fileObject)
        if os.path.isfile(fileObjectPath) or os.path.islink(fileObjectPath):
            os.unlink(fileObjectPath)
        else:
            shutil.rmtree(fileObjectPath)
def deleteFolder(folderPath):
    folderPath = removeLastSlash(folderPath)
    filesInDir = os.listdir(folderPath)
    for file in filesInDir:  # loop to delete each file in folder
        os.remove(f'{folderPath}/{file}')
    os.rmdir(folderPath)
def removeLastSlash(folderPath):
    while folderPath.endswith("/") or folderPath.endswith("\\"):
        folderPath = folderPath[:-1]
    return folderPath
