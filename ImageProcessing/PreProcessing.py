
import bm4d
import cv2
import numpy as np
import skimage.feature
from matplotlib import pyplot as plt

from skimage import morphology, img_as_float, img_as_ubyte, filters

import bm3d
from skimage.morphology import disk


def BinaryErosion(img1,img2,img3,img4):

    # # Taking a matrix of size 5 as the kernel
    # kernel = np.ones((3, 3), np.uint8)
    #
    # # The first parameter is the original image,
    # # kernel is the matrix with which image is
    # # convolved and third parameter is the number
    # # of iterations, which will determine how much
    # # you want to erode/dilate a given image.
    # #im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_erosion = cv2.erode(img1, kernel, iterations=1)
    # cv2.imshow('Erosion', img_erosion)
    # img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
    # cv2.imshow('Input', img1)
    # blur = cv2.GaussianBlur(img1, (3, 3), 0)
    # ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow('otsu', th3)
    # img_dilation = cv2.dilate(th3, kernel, iterations=1)
    # # cv2.imshow('dilation', img_dilation)
    # median = cv2.medianBlur(img1, 3)
    # # cv2.imshow('median', median)
    # img_removed= morphology.remove_small_objects(img1, min_size=1000)
    # thresh = threshold_isodata(img1)
    #
    # binary = (img1 > thresh)
    # binary= binary.astype(np.uint8)
    # binary*=255
    # # blur = cv2.GaussianBlur(binary, (5, 5), 0)
    # # img_dilation = cv2.dilate(binary, kernel, iterations=1)
    # cv2.imshow('isodata', binary)
    imgf1= img_as_float(img1)

    denoise=bm3d.bm3d(imgf1,sigma_psd=0.2,stage_arg=bm3d.BM3DStages.ALL_STAGES)

    cv2.imshow('denoise', denoise)
    cv2.waitKey(0)

def bm3dDenoise(img1):

    imgf1= img_as_float(img1)

    denoise=bm3d.bm3d(imgf1,sigma_psd=0.2,stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)

    cv2.imshow('bm3d denoise 1', denoise)
    #cv2.waitKey(0)

def bm4dDenoise(img1, img2, img3):

    imgf1= img_as_float(img1)
    imgf2 = img_as_float(img2)
    imgf3 = img_as_float(img3)

    denoise2=bm4d.bm4d(imgf2, sigma_psd=0.2)
    cv2.imshow('bm4d denoise 1', denoise2)
    imgtotal=(imgf1+imgf2+imgf3)/3
    cv2.imshow('imtotal', imgtotal)
    denoisetotal = bm4d.bm4d(imgtotal, sigma_psd=0.2)
    cv2.imshow('bm4d denoise 3', denoisetotal)

    #cv2.waitKey(0)

def bm4dDenoiseAdap(img2):

    imgf2 = img_as_float(img2)
    denoise2 = bm4d.bm4d(imgf2, sigma_psd=0.2)
    cv2.imshow('bm4d denoise 1', denoise2)
    img = img_as_ubyte(denoise2)
    thresh1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 3, 5)
    cv2.imshow('bm4d denoise 3 Adap', thresh1)
    # cv2.waitKey(0)

def bm4dDenoise2(img0, img1, img2, img3, img4):
    imgf0 = img_as_float(img0)
    imgf1 = img_as_float(img1)
    imgf2 = img_as_float(img2)
    imgf3 = img_as_float(img3)
    imgf4 = img_as_float(img4)

    #denoise2 = bm4d.bm4d(imgf2, sigma_psd=0.2)
    #cv2.imshow('denoise2', denoise2)
    imgtotal = (imgf0 + imgf1 + imgf2 + imgf3+ imgf4) / 5
    cv2.imshow('imtotal', imgtotal)
    denoisetotal = bm4d.bm4d(imgtotal, sigma_psd=0.2)
    cv2.imshow('bm4d denoise 5', denoisetotal)
    # cv2.waitKey(0)
def bm4dDenoise3(img0, img1, img2, img3, img4):
    imgf0 = img_as_float(img0)
    imgf1 = img_as_float(img1)
    imgf2 = img_as_float(img2)
    imgf3 = img_as_float(img3)
    imgf4 = img_as_float(img4)

    #denoise2 = bm4d.bm4d(imgf2, sigma_psd=0.2)
    #cv2.imshow('denoise2', denoise2)
    imgtotal = (imgf0 + imgf1 + imgf2 + imgf3+ imgf4) / 5
    imgmull= (imgtotal * imgf2) /img2;

    cv2.imshow('imgmull', imgmull)
    denoisetotal = bm4d.bm4d(imgmull, sigma_psd=0.2)
    cv2.imshow('bm4d denoise 5 2', denoisetotal)
    denoisetotaltre = (0.5<denoisetotal)*imgmull
    cv2.imshow('bm4d denoise 5 2 tresh', denoisetotaltre)
def sobel(img):


    # resize image
    img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_CUBIC)

    # convert image to gray scale image
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply laplacian blur
    laplacian = cv2.Laplacian(img, cv2.CV_64F)

    # sobel x filter where dx=1 and dy=0
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=7)

    # sobel y filter where dx=0 and dy=1
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=7)

    # combine sobel x and y
    sobel = cv2.bitwise_and(sobelx, sobely)
    # plot images
    cv2.imshow('Laplacian', laplacian)
    cv2.imshow('SobelX', sobelx)
    cv2.imshow('SobelY', sobely)
    cv2.imshow('Sobel', sobel)


def sobel2(img):


    # resize image
    #img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_CUBIC)
    imgf = img_as_float(img)
    imgbm4d = bm4d.bm4d(imgf, sigma_psd=0.2)
    cv2.imshow('bm4d denoise 1', imgbm4d)

    # apply laplacian blur
    laplacian = cv2.Laplacian(imgbm4d, cv2.CV_64F)

    # sobel x filter where dx=1 and dy=0
    sobelx = cv2.Sobel(imgbm4d, cv2.CV_64F, 1, 0, ksize=7)

    # sobel y filter where dx=0 and dy=1
    sobely = cv2.Sobel(imgbm4d, cv2.CV_64F, 0, 1, ksize=7)

    # combine sobel x and y
    sobel = cv2.bitwise_and(sobelx, sobely)
    # plot images
    cv2.imshow('Laplacian', laplacian)
    cv2.imshow('SobelX', sobelx)
    cv2.imshow('SobelY', sobely)
    cv2.imshow('Sobel', sobel)

def tophat(img):
    imgf = img_as_float(img)
    imgbm4d = bm4d.bm4d(imgf, sigma_psd=0.2)
    cv2.imshow('bm4d denoise 1', imgbm4d)
    # Defining the kernel to be used in Top-Hat
    filterSize = (7, 7)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       filterSize)

    # Reading the image named 'input.jpg'

    input_image = imgbm4d

    # Applying the Black-Hat operation
    tophat_img = cv2.morphologyEx(input_image,
                                  cv2.MORPH_BLACKHAT,
                                  kernel)

    cv2.imshow("original", input_image)
    cv2.imshow("tophat", tophat_img)
def erosiondilation(img):

    imgf = img_as_float(img)
    imgbm4d = bm4d.bm4d(imgf, sigma_psd=0.2)
    cv2.imshow('bm4d denoise 1', imgbm4d)

    kernel = np.ones((5, 5), np.uint8)

    # The first parameter is the original image,
    # kernel is the matrix with which image is
    # convolved and third parameter is the number
    # of iterations, which will determine how much
    # you want to erode/dilate a given image.
    img_erosion = cv2.erode(imgbm4d, kernel, iterations=1)
    img_dilation = cv2.dilate(imgbm4d, kernel, iterations=1)


    cv2.imshow('Erosion', img_erosion)
    cv2.imshow('Dilation', img_dilation)
def sharpen(img0, img1, img2):
    kernel = np.array([[0, -1, 0],
                       [-1, 7, -1],
                       [0, -1, 0]])
    image_sharp = cv2.filter2D(src=img1, ddepth=-1, kernel=kernel)
    cv2.imshow('AV CV- Winter Wonder Sharpened1', image_sharp)

    imgf = img_as_float(image_sharp)
    imgbm4d = bm4d.bm4d(imgf, sigma_psd=0.2)
    cv2.imshow('bm4d denoise 1', imgbm4d)

    imgb = (imgbm4d * 255).astype(np.ubyte)


    scharr=filters.scharr(imgb)
    cv2.imshow('scharr', scharr)

    blur = cv2.GaussianBlur(imgb, (5, 5), 0)
    cv2.imshow('blur', blur)

    ret3, th3 = cv2.threshold(imgb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('otsu', th3)

    # imgb= img_as_ubyte(imgb)

    thresh2 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 199, 5)
    cv2.imshow('adaptive', thresh2)

    edge_sobel = filters.sobel(imgbm4d)

    cv2.imshow('edge_sobel', edge_sobel)
    img0f= img_as_float(img0);
    mig2f= img_as_float(img2);
    edge_sobel2= ((edge_sobel*img0f) *mig2f)/edge_sobel
    cv2.imshow('edge_sobel2', edge_sobel2)

    thresh2 = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)

    kernel = np.ones((3, 3), np.uint8)
    img_dilation = cv2.dilate(th3, kernel, iterations=1)


    cv2.imshow('img_dilation', img_dilation)
def cannyEdge(img):
    imgf = img_as_float(img);
    edges2 = skimage.feature.canny(img, sigma=3)
    edges2= 1*edges2
    imgf = img_as_float(edges2);
    cv2.imshow('cannyedge', imgf)

def andform(img0, img1, img2, img3, img4):
    kernel = np.array([[0, -1, 0],
                       [-1, 7, -1],
                       [0, -1, 0]])
    image_sharp = cv2.filter2D(src=img2, ddepth=-1, kernel=kernel)
    imgf = img_as_float(image_sharp)
    imgbm4d = bm4d.bm4d(imgf, sigma_psd=0.2)
    imgb = (imgbm4d * 255).astype(np.ubyte)

    filtered = filters.meijering(imgb, sigmas=[1], black_ridges=False)

    cv2.imshow('filtered', filtered)


    #cv2.imshow('denoise2', denoise2)


from PIL import Image

img = Image.open('skeleton_red.tif')
rgba = img.convert("RGBA")
datas = rgba.getdata()

newData = []
for item in datas:
    if item[0] == 0 and item[1] == 0 and item[2] == 0:  # finding black colour by its RGB value
        # storing a transparent value when we find a black colour
        newData.append((255, 255, 255, 0))
    else:
        newData.append(item)  # other colours remain unchanged

rgba.putdata(newData)
rgba.save("transparent_skeleton_red.png", "PNG")

img0 = cv2.imread('..\images\HFHS13-16wk_HFHS13-4z_z07.tif', 0)
img1 = cv2.imread('..\images\HFHS13-16wk_HFHS13-4z_z08.tif', 0)
img2 = cv2.imread('..\images\HFHS13-16wk_HFHS13-4z_z09.tif', 0)
img3 = cv2.imread('..\images\HFHS13-16wk_HFHS13-4z_z10.tif', 0)
img4 = cv2.imread('..\images\HFHS13-16wk_HFHS13-4z_z11.tif', 0)

img2 = cv2.imread('Img1\img3.PNG', 0)
name="img3_"
#img2 = cv2.imread('..\images\HFHS13-16wk_HFHS13-4z_z11.tif', 0)
cv2.imshow('main img2', img2)


imgf = img_as_float(img2)
imgbm4d = bm4d.bm4d(imgf, sigma_psd=0.1)
cv2.imshow(name+'bm4d', imgbm4d)

imgb = ((imgbm4d - imgbm4d.min()) * (1/(imgbm4d.max() - imgbm4d.min()) * 255)).astype('uint8')
skimage.io.imsave(name+"bm4d.tiff",imgb)
#= img_as_float(imgb)


newarr =imgb.reshape(512,512)
cv2.imshow(name+'newarr', newarr)

kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
image_sharp = cv2.filter2D(src=newarr, ddepth=-1, kernel=kernel)
cv2.imshow('AV CV- Winter Wonder Sharpened1', image_sharp)
skimage.io.imsave(name+"image_sharp.tiff",image_sharp)

image_gausian= skimage.filters.gaussian(image_sharp, sigma=1)
image_gausian_uint8 = (image_gausian * 255).astype('uint8')
cv2.imshow(name+'gausian', image_gausian)
skimage.io.imsave(name+"image_gausian1.tiff",image_gausian_uint8)

# fig, ax=skimage.filters.try_all_threshold(image_gausian, figsize=(10,10), verbose=True)
# fig_nums = plt.get_fignums()
# figs = [plt.figure(n) for n in fig_nums]
# for i in range(0,len(figs)):
#     figs[i].savefig(ax[i].title.get_text()+".png")
# for item in ax:
#     print(item.title.get_text())
#     k=plt.figure(3)
#     k.savefig(item.title.get_text()+".png")
#     item.plot().get_figure().savefig( item.title.get_text()+".png")
# plt.show()
# isodata
# li
# mean
# minimum
# otsu
# triangle
# yen
isodata_tresh=skimage.filters.threshold_isodata(image_gausian)
isodata_uint8 = ((image_gausian > isodata_tresh) * 255).astype('uint8')
#cv2.imshow('triange treshhold', isodata_uint8)
#skimage.io.imsave(name+"tresh_isodata.tiff",isodata_uint8)

li_tresh=skimage.filters.threshold_li(image_gausian)
li_uint8 = ((image_gausian > li_tresh) * 255).astype('uint8')
#cv2.imshow('triange treshhold', li_uint8)
#skimage.io.imsave(name+"tresh_li.tiff",li_uint8)

mean_tresh=skimage.filters.threshold_mean(image_gausian)
mean_uint8 = ((image_gausian > mean_tresh) * 255).astype('uint8')
#cv2.imshow('triange treshhold', mean_uint8)
#skimage.io.imsave(name+"tresh_mean.tiff",mean_uint8)

minimum_tresh=skimage.filters.threshold_minimum(image_gausian)
minimum_uint8 = ((image_gausian > minimum_tresh) * 255).astype('uint8')
#cv2.imshow('triange treshhold', minimum_uint8)
#skimage.io.imsave(name+"tresh_minimum.tiff",minimum_uint8)

otsu_tresh=skimage.filters.threshold_otsu(image_gausian)
otsu_uint8 = ((image_gausian > otsu_tresh) * 255).astype('uint8')
#cv2.imshow('triange treshhold', minimum_uint8)
#skimage.io.imsave(name+"tresh_otsu.tiff",otsu_uint8)

yen_tresh=skimage.filters.threshold_yen(image_gausian)
yen_uint8 = ((image_gausian > yen_tresh) * 255).astype('uint8')
#cv2.imshow('triange treshhold', minimum_uint8)
#skimage.io.imsave(name+"tresh_yen.tiff",yen_uint8)

triangle_tresh=skimage.filters.threshold_triangle(image_gausian)
triangle_uint8 = ((image_gausian > triangle_tresh) * 255).astype('uint8')
cv2.imshow('triange treshhold', triangle_uint8)
#skimage.io.imsave(name+"tresh_triangle.tiff",triangle_uint8)

triang_bool= 0<triangle_uint8

triang_bool = cv2.imread('Img1\ATTENTION_UNET\img3.PNG', 0)
triang_bool= 0<triang_bool


# footprint = disk(2)
# binary_closed = skimage.morphology.binary_closing(triang_bool,footprint)
# binary_closed_uint8 = (binary_closed * 255).astype('uint8')
# cv2.imshow('binary_closed_uint8', binary_closed_uint8)
# skimage.io.imsave(name+"binary_closed_uint8.tiff",binary_closed_uint8)
#
# remove_small_objects = morphology.remove_small_objects(binary_closed, 10,connectivity=2)
# remove_small_object_uint8 = (remove_small_objects * 255).astype('uint8')
# cv2.imshow(name+'remove_small_object_uint8', remove_small_object_uint8)

# removesmallholes_bool = morphology.remove_small_holes(remove_small_objects, 10, connectivity=2)
# removesmallholes_uint8 = (removesmallholes_bool * 255).astype('uint8')
# cv2.imshow('removesmallholes', removesmallholes_uint8)

# binary_dilation= skimage.morphology.binary_dilation(remove_small_objects)
# dilation_uint8 = (binary_dilation * 255).astype('uint8')
# cv2.imshow('dilation_uint8', dilation_uint8)
# skimage.io.imsave("dilation_uint8.tiff",dilation_uint8)

# oppening= skimage.morphology.opening(binary_dilation, morphology.square(3))
# oppening_uint8 = (oppening * 255).astype('uint8')
# cv2.imshow('oppening', oppening_uint8)
# skimage.io.imsave("oppening.tiff",oppening_uint8)

skeleton_bool = skimage.morphology.skeletonize(triang_bool)
skeleton_uint8 = (skeleton_bool * 255).astype('uint8')
cv2.imshow('skeleton_uint8', skeleton_uint8)
skimage.io.imsave(name+"skeleton.tiff",skeleton_uint8)
#skimage.io.imsave(name+"remove_small_object_uint8.tiff",remove_small_object_uint8)

#imgfra=skimage.filters.frangi(newarr,black_ridges=False)
#new_arr = ((imgfra - imgfra.min()) * (1/(imgfra.max() - imgfra.min()) * 255)).astype('uint8')
#cv2.imshow('Frangi2', new_arr)
#sobel(img2)
#sobel2(img2)
#tophat(img2)
#erosiondilation(img2)
#sharpen(img1, img2, img3)
#cannyEdge(img2)
#dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(img2))
#plt.figure(num=None, figsize=(8, 6), dpi=80)
#plt.imshow(np.log(abs(dark_image_grey_fourier)), cmap='gray')

# img_float32 = np.float32(img2)
# dft = cv2.dft(img_float32, flags = cv2.dft_complex_output)
# dft_shift = np.fft.fftshift(dft)
# magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
#
# plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
# plt.title('magnitude spectrum'), plt.xticks([]), plt.yticks([])
#
# plt.show()

#cv2.imshow('spechtrum', dark_image_grey_fourier)
#andform(img0,img1,img2,img3,img4)
#bm3dDenoise(img2)
#bm4dDenoise(img1,img2,img3)

#bm4dDenoiseAdap(img2)
#bm4dDenoise2(img0,img1,img2,img3,img4)
#bm4dDenoise3(img0,img1,img2,img3,img4)
cv2.waitKey(0)