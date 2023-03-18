import os
import cv2
from plantcv.plantcv import dilate
from plantcv.plantcv import params
from plantcv.plantcv import find_objects
from plantcv.plantcv import color_palette
from plantcv.plantcv import image_subtract
from plantcv.plantcv.morphology import find_branch_pts
from plantcv.plantcv._debug import _debug


def segment_skeleton1(skel_img):
    """Segment a skeleton image into pieces.
    Inputs:
    skel_img         = Skeletonized image
    mask             = (Optional) binary mask for debugging. If provided, debug image will be overlaid on the mask.
    Returns:
    segmented_img       = Segmented debugging image
    segment_objects     = list of contours
    :param skel_img: numpy.ndarray
    :param mask: numpy.ndarray
    :return segmented_img: numpy.ndarray
    :return segment_objects: list
    """
    # Store debug
    debug = params.debug
    params.debug = None

    # Find branch points
    bp = find_branch_pts(skel_img)

    bp = dilate(bp, 3, 1)

    # Subtract from the skeleton so that leaves are no longer connected
    segments = image_subtract(skel_img, bp)

    # Gather contours of leaves
    segment_objects, _ = find_objects(segments, segments)

    # Reset debug mode
    params.debug = debug




    return segment_objects