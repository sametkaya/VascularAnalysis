import bm4d
import cv2
import numpy as np
import skimage
from matplotlib import pyplot as plt
from skimage import img_as_float
from skimage.filters.ridges import frangi
from skimage.restoration import estimate_sigma, denoise_nl_means

from frangiFilter2D import FrangiFilter2D, eig2image

img2 = cv2.imread('..\images\HFHS13-16wk_HFHS13-4z_z09.tif')

sigma_est= np.mean(estimate_sigma(img2,multichannel=False))

denoised= denoise_nl_means(img2,h=1.2*sigma_est,fast_mode=True, patch_size=5, patch_distance=3, multichannel=False)

cv2.imshow('main denoised ', denoised)

cv2.imshow('main img2 ', img2)
gray_image = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray_image)
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
image_sharp = cv2.filter2D(src=gray_image, ddepth=-1, kernel=kernel)
cv2.imshow('AV CV- Winter Wonder Sharpened1', image_sharp)

imgf = img_as_float(image_sharp)
imgbm4d = bm4d.bm4d(imgf, sigma_psd=0.2)
imgb = (imgbm4d * 255).astype(np.ubyte)
cv2.imshow("denoised", imgb)
#= img_as_float(imgb)
newarr =imgb.reshape(512,512)

frangi_result1 = skimage.filters.ridges.frangi(newarr,scale_range=(1, 10), scale_step=1,mode='nearest',black_ridges=False)
#imgr = (frangi_result1 * 255).astype(np.ubyte)
#cv2.imshow('imgr ', imgr)
plt.imshow(frangi_result1, cmap = 'gray')
plt.show()

frangi_resultGray = skimage.filters.ridges.frangi(image_sharp,scale_range=(1, 20), scale_step=3,mode='reflect',black_ridges=False)

plt.imshow(frangi_resultGray, cmap = 'gray')
plt.show()


#plt.subplot(1),plt.imshow(frangi_result, cmap = 'gray')
#plt.title('magnitude spectrum')
#plt.show()
cv2.waitKey(0)