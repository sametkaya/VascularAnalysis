import sys
from collections import deque


import numpy as np
from pip._internal.cli.spinners import hidden_cursor


def find_path_tip_to_branch(skel_img, start_point):
    '''
    img_binary: skeleton image

    '''
    height, width = skel_img.shape

    if(start_point < (0, 0) and start_point > (width, height) ):
        return None

    skel_img_padded = np.pad(skel_img, pad_width=1)


    # All 8 directions
    delta = [(-1, -1),  (-1, 0),    (-1, 1),
             (0, -1),               (0, 1),
             (1, -1),   (1, 0),     (1, 1)]


    # The actual BFS algorithm
    start_point_padded= (start_point[0]+1, start_point[1]+1)
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

        if hit_count == 1:
            bfs.append(next_yx)
            prev_point = path_points[-1]
            path_points = np.append(path_points, [next_yx], axis=0)
        else:
            break

    path_points = path_points - 1

    return path_points