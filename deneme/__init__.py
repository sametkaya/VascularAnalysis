import cv2

from .hessian import Hessian2D
from .frangiFilter2D import FrangiFilter2D, eig2image

img2 = cv2.imread('..\images\HFHS13-16wk_HFHS13-4z_z09.tif', 0)
cv2.imshow('main img2 ', img2)
imgr,o,i=FrangiFilter2D(img2)
cv2.imshow('imgr ', imgr)

cv2.waitKey(0)