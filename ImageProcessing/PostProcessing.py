
import cv2
import numpy
import skimage
from skimage import img_as_float
import pandas as pd

from ImageProcessing.PostProcess.find_all_branch_paths import find_all_branch_paths
from ImageProcessing.PostProcess.find_all_branch_paths2 import find_all_branch_paths2
from ImageProcessing.PostProcess.find_branch_pts import find_branch_pts
from ImageProcessing.PostProcess.find_path_tip_to_branch import find_path_tip_to_branch
from ImageProcessing.PostProcess.find_tips import find_tips
from ImageProcessing.PostProcess.segment_skeleton import segment_skeleton

from plantcv import plantcv as pcv

#skeleton = cv2.imread('skeleton1.tiff', 0)
skeleton_uint8 = cv2.imread('Img1\PreProcess\img3_skeleton.tiff', 0)
#skeleton = cv2.imread('Img1\After_Segmented_PreProcess\img3_skeleton_notremoved.tiff', 0)
name="img3_"
#imgf = img_as_float(skeleton_uint8)
#skeleton_uint8=imgf.astype('uint8')

#pruned_skeleton, segmented_img, segment_objects = pcv.morphology.prune(skel_img=skeleton_uint8, size=150)
#skimage.io.imsave(r"Img1\PreProcess\segmented_img.tiff",segmented_img)

segmented_img, segmented_obj=segment_skeleton(skeleton_uint8)

skimage.io.imsave(r"Img1\PreProcess\segmented_img1.png",segmented_img)
cv2.imwrite(r"Img1\PreProcess\segmented_img11.png",segmented_img)

labeled_img,veinlenght = pcv.morphology.segment_path_length(segmented_img=segmented_img,objects=segmented_obj, label="default")
df = pd.DataFrame(veinlenght)
df.to_csv(r'Img1\PreProcess\lenghts.csv', index=False, header=False)
skimage.io.imsave(r"Img1\PreProcess\segmented_lenght_img1.tiff",labeled_img)

#tips_img,tips_dict= find_tips(skeleton_uint8)
#skimage.io.imsave(r"Img1\PreProcess\tips_img.tiff", tips_img)

branch_paths = find_all_branch_paths2(skeleton_uint8)

df = pd.DataFrame(branch_paths)
df.to_csv(r'Img1\PreProcess\brachpoints3.csv', index=False, header=False)
#path=find_path_tip_to_branch(skeleton_uint8, (405, 172), (412, 177))

#skimage.io.imsave(name+"tips_img.tiff",tips_img)
branch_points_img, branch_points_dict = find_branch_pts(skel_img=skeleton_uint8)
#numpy.savetxt("\Img1\PreProces\points.txt", branch_points_dict, delimiter=',')
blist = [(k, v) for k, v in branch_points_dict.items()]

df = pd.DataFrame(blist)
df.to_csv(r'Img1\PreProcess\brachpoints.csv', index=False, header=False)
#skimage.io.imsave("Img1\PreProcess\skeleton_brach.tiff",branch_points_img)
clist=list(branch_points_dict.values())
branch_points_cicle_img = cv2.cvtColor(branch_points_img,cv2.COLOR_GRAY2RGB)
branch_points_cicle_on_vein_img = cv2.cvtColor(skeleton_uint8,cv2.COLOR_GRAY2RGB)
branch_points_cicle_point_on_vein_img = cv2.cvtColor(skeleton_uint8,cv2.COLOR_GRAY2RGB)

#branch_points_cicle_img = cv2.circle(branch_points_img, clist[0], 10, (255, 0, 0), 1)
for center in clist:
    branch_points_cicle_img=cv2.circle(branch_points_cicle_img, center, 10, (255, 0, 0), 1)
    branch_points_cicle_on_vein_img = cv2.circle(branch_points_cicle_on_vein_img, center, 10, (255, 0, 0), 1)
    branch_points_cicle_point_on_vein_img =cv2.circle(branch_points_cicle_point_on_vein_img, center, 2, (0, 255, 0), 2)
    branch_points_cicle_point_on_vein_img = cv2.circle(branch_points_cicle_point_on_vein_img, center, 10, (255, 0, 0), 1)
skimage.io.imsave(r"Img1\PreProcess\branch_points_cicle_img.tiff",branch_points_cicle_img)
skimage.io.imsave(r"Img1\PreProcess\branch_points_cicle_on_vein_img.tiff",branch_points_cicle_on_vein_img)
skimage.io.imsave(r"Img1\PreProcess\branch_points_cicle_point_on_vein_img.tiff",branch_points_cicle_point_on_vein_img)

tlist=list(tips_dict.values())
tips_point_image=cv2.cvtColor(tips_img,cv2.COLOR_GRAY2RGB)
tips_points_cicle_point_on_vein_img = cv2.cvtColor(skeleton_uint8,cv2.COLOR_GRAY2RGB)
for center in tlist:
    tips_point_image = cv2.circle(tips_point_image, center, 10, (255, 0, 0), 1)
    tips_points_cicle_point_on_vein_img = cv2.circle(tips_points_cicle_point_on_vein_img, center, 2, (0, 255, 0), 2)
    tips_points_cicle_point_on_vein_img = cv2.circle(tips_points_cicle_point_on_vein_img, center, 10, (255, 0, 0), 1)

skimage.io.imsave(r"Img1\PreProcess\tips_point_image.tiff",tips_point_image)
skimage.io.imsave(r"Img1\PreProcess\tips_points_cicle_point_on_vein_img.tiff",tips_points_cicle_point_on_vein_img)