
import numpy as np
from ImageProcessing.PostProcess.find_tips import find_tips
from collections import deque

def find_all_branch_paths(skel_img):

    branch_paths = {}

    height, width = skel_img.shape

    skel_img_padded = np.pad(skel_img, pad_width=1)

    # All 8 directions
    delta = [(-1, -1), (-1, 0), (-1, 1),
             (0, -1),           (0, 1),
             (1, -1),   (1, 0), (1, 1)]
    index = 0

    tip_dict = find_tips(skel_img_padded)
    for tip in tip_dict.values():
        # The actual BFS algorithm
        #start_point = tip
        start_point_padded = (tip[0], tip[1])
        if skel_img_padded[tip[0]][tip[1]] == 0:
            continue
        bfs = deque([start_point_padded])
        path_points = np.array([start_point_padded])
        found = False

        prev_point = [-1, -1]
        while len(bfs) > 0:
            y, x = bfs.popleft()
            # print(y,x)

            # Look all 8 directions for a good path
            hit_count = 0
            next_yx = (x, y)

            for dy, dx in delta:
                yy, xx = y + dy, x + dx
                # If the next position hasn't already been looked at and it's white
                if not (yy == prev_point[0] and xx == prev_point[1]) and skel_img_padded[yy][xx] > 0:
                    hit_count += 1
                    next_yx = (yy, xx)
            # 0 hit means line end without branch point
            # 2 or more hit means line end with branch point
            if hit_count == 1:
                bfs.append(next_yx)
                skel_img_padded[prev_point[0]][prev_point[1]] = 0
                prev_point = path_points[-1]
                path_points = np.append(path_points, [next_yx], axis=0)
            elif hit_count == 0:
                skel_img_padded[prev_point[0]][prev_point[1]] = 0
                prev_point = path_points[-1]
                skel_img_padded[prev_point[0]][prev_point[1]] = 0
                break
            else:
                #skel_img_padded[prev_point[0]][prev_point[1]] = 0
                break

        path_points = path_points - 1
        branch_paths[index] = path_points
        index += 1



    return branch_paths