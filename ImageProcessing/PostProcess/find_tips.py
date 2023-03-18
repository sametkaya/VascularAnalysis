# Find tips from skeleton image

import os
import cv2
import numpy as np
from plantcv.plantcv import params
from plantcv.plantcv import dilate
from plantcv.plantcv import outputs
from plantcv.plantcv import find_objects
from plantcv.plantcv._debug import _debug


def find_tips(skel_img):
    """Find tips in skeletonized image.
    The endpoints algorithm was inspired by Jean-Patrick Pommier: https://gist.github.com/jeanpat/5712699
    Inputs:
    skel_img    = Skeletonized image
    mask        = (Optional) binary mask for debugging. If provided, debug image will be overlaid on the mask.
    label        = optional label parameter, modifies the variable name of observations recorded
    Returns:
    tip_img   = Image with just tips, rest 0
    :param skel_img: numpy.ndarray
    :param mask: numpy.ndarray
    :param label: str
    :return tip_dict-> [index,[[point],[neighbors]]]
    """
    # In a kernel: 1 values line up with 255s, -1s line up with 0s, and 0s correspond to dont care
    endpoint1 = np.array([[-1, -1, -1],
                          [-1,  1, -1],
                          [ 0,  1,  0]])
    endpoint2 = np.array([[-1, -1, -1],
                          [-1,  1,  0],
                          [-1,  0,  1]])

    endpoint3 = np.rot90(endpoint1)
    endpoint4 = np.rot90(endpoint2)
    endpoint5 = np.rot90(endpoint3)
    endpoint6 = np.rot90(endpoint4)
    endpoint7 = np.rot90(endpoint5)
    endpoint8 = np.rot90(endpoint6)

    endpoints = [endpoint1, endpoint2, endpoint3, endpoint4, endpoint5, endpoint6, endpoint7, endpoint8]
    skel_img_padded = np.pad(skel_img, pad_width=1)
    tip_img = np.zeros(skel_img_padded.shape[:2], dtype=int)
    #skel_img_padded = np.pad(skel_img, pad_width=1)
    for endpoint in endpoints:
        tip_img = np.logical_or(cv2.morphologyEx(skel_img_padded, op=cv2.MORPH_HITMISS, kernel=endpoint, borderType=cv2.BORDER_CONSTANT, borderValue=0), tip_img)
    tip_img = np.delete(tip_img, 0, 0)
    tip_img = np.delete(tip_img, 0, 1)
    tip_img = np.delete(tip_img, tip_img.shape[0]-1, 0)
    tip_img = np.delete(tip_img, tip_img.shape[1]-1, 1)
    tip_img = tip_img.astype(np.uint8) * 255


    tip_objects, _ = find_objects(tip_img, tip_img)
    # All 8 directions
    delta = [(-1, -1), (-1, 0), (-1, 1),
             (0, -1), (0, 1),
             (1, -1), (1, 0), (1, 1)]
    tip_dict = {}
    for i, tip in enumerate(tip_objects):
        x, y = tip.ravel()[:2]
        coord = [int(y), int(x)]
        neigbors = []
        for dy, dx in delta:
            yy, xx = y + dy, x + dx
            # If the next position hasn't already been looked at and it's white
            if skel_img_padded[yy][xx] > 0:
                neigbors.append([yy, xx])
        tip_dict[i] = [coord, neigbors]



    return tip_dict
