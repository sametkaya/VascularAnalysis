import numpy as np
import PIL as pil
import bm3d
import os
def bm3d_denoise(img_uint8):
    img_float64 = img_uint8.astype(np.float64) / img_uint8.max()
    img_bm3d = bm3d.bm3d(img_float64, sigma_psd=0.1)
    img_bm3d_uint8 = ((img_bm3d - img_bm3d.min()) * (1 / (img_bm3d.max() - img_bm3d.min()) * 255)).astype('uint8')
    return img_bm3d_uint8

def load_images_from_folder(folder):
    image3d = np.zeros((100, 512, 512), dtype=np.uint8)
    #img = np.asarray(pil.Image.open(os.path.join(folder, "HFHS13-16wk_HFHS13-4z_z12.tif")).convert('L'))
    #for z in range(100):
    #    image3d[z] = img

    for index, filename in enumerate(os.listdir(folder)):
        img = np.asarray(pil.Image.open(os.path.join(folder,filename)).convert('L'))
        if img is not None:
            image3d[index]=img

    return image3d

image3d=load_images_from_folder("..\\images\\")
print(image3d.shape)
image3dz = np.zeros((100, 512, 512), dtype=np.uint8)
image3dx = np.zeros((100, 512, 512), dtype=np.uint8)
image3dy = np.zeros((100, 512, 512), dtype=np.uint8)
for i in range(100):
    im_z = image3d[i,0:,0:]
    img_z_bm3d_uint8 = bm3d_denoise(im_z)
    im = pil.Image.fromarray(img_z_bm3d_uint8)
    im.save("z_"+str(i)+"_im.tiff")

for i in range(512):
    im_x=np.transpose(image3d[0:,0:,i])
    img_x_bm3d_uint8=bm3d_denoise(im_x)
    im = pil.Image.fromarray(img_x_bm3d_uint8)
    im.save("x_"+str(i)+"_im.tiff")

    im_y=image3d[0:,i,0:]
    img_y_bm3d_uint8 = bm3d_denoise(im_y)
    im = pil.Image.fromarray(img_y_bm3d_uint8)
    im.save("y_"+str(i)+"_im.tiff")



# arr = np.array([[[1, 3, 5],
#                  [1, 3, 5],
#                  [1, 3, 5]],
#                 [[2, 4, 6],
#                  [2, 4, 6],
#                  [2, 4, 6]],
#                 [[3, 5, 7],
#                  [3, 5, 7],
#                  [3, 5, 7]],
#                 [[4, 6, 8],
#                  [4, 6, 8],
#                  [4, 6, 8]]])
#
# ay = arr[0:,0,0:]
# ax = np.transpose(arr[0:,0:,0])
# #arrs= ax.reshape(4,3,3)
# az = arr[0,0:,0:]
i=0

