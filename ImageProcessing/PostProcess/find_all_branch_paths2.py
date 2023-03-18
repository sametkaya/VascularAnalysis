import cv2
import numpy as np
from ImageProcessing.PostProcess.find_tips import find_tips
from collections import deque

from ImageProcessing.PostProcess.segment_skeleton1 import segment_skeleton1


def find_all_branch_paths2(skel_img):
    branch_list=segment_skeleton1(skel_img)
    branch_paths = []

    height, width = skel_img.shape

    skel_img_padded = np.pad(skel_img, pad_width=1)

    # All 8 directions
    # delta = [(-1, -1), (-1, 0), (-1, 1),
    #          (0, -1),           (0, 1),
    #          (1, -1),   (1, 0), (1, 1)]
    delta = [(-1, -1),  (-1, 1), (1, -1),  (1, 1), (-1, 0), (0, -1), (0, 1), (1, 0)]
    index = 0

    for branch in branch_list:
        path_points = branch.reshape(-1, 2)
        mid = int((path_points.shape[0] + 1) / 2) + 1
        path_points = path_points[:mid]
        for tip_no in range(2):
            start_point_padded = []
            prev_point = [-1, -1]
            if tip_no == 0: # start tip
                start_point_padded = path_points[0]+1

                if len(path_points)==1:
                    prev_point = path_points[0]+1
                else:
                    prev_point = path_points[1] + 1
            else: # end tip
                start_point_padded = path_points[-1]+1
                if len(path_points) == 1:
                    prev_point = path_points[-2] + 1
                else:
                    prev_point = path_points[-2] + 1


            bfs = deque([start_point_padded])

            while len(bfs) > 0:
                x, y = bfs.popleft()
                # print(y,x)

                # Look all 8 directions for a good path
                hit_count = 0
                next_yx = np.array([[x, y]])-1

                for dy, dx in delta:
                    yy, xx = y + dy, x + dx
                    # If the next position hasn't already been looked at and it's white
                    if not (xx == prev_point[0] and yy == prev_point[1]) and skel_img_padded[yy][xx] > 0:
                        hit_count += 1
                        next_yx = np.array([[xx, yy]])-1
                # 0 hit means line end without branch point
                # 2 or more hit means line end with branch point
                if hit_count == 1:
                    bfs.append(next_yx[0]+1)
                    if tip_no == 0:
                        prev_point = path_points[0]+1
                        path_points = np.insert(path_points, 0, next_yx, axis=0)
                    else:
                        prev_point = path_points[-1]+1
                        path_points = np.append(path_points, next_yx, axis=0)
                elif hit_count > 1:
                    # if tip_no == 0:
                    #     path_points = np.insert(path_points, 0, next_yx, axis=0)
                    # else:
                    #     path_points = np.append(path_points, next_yx, axis=0)

                    # skel_img_padded[prev_point[0]][prev_point[1]] = 0
                    # prev_point = path_points[-1]
                    # skel_img_padded[prev_point[0]][prev_point[1]] = 0
                    break
                else:
                    # skel_img_padded[prev_point[0]][prev_point[1]] = 0
                    break
        lenght = float(cv2.arcLength(path_points, False))
        branch_paths.append([index, lenght, path_points[0], path_points[-1], path_points])

        index += 1




    return branch_paths