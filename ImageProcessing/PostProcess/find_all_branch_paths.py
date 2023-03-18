
import numpy as np

from ImageProcessing.PostProcess.find_branch_pts import find_branch_pts
from ImageProcessing.PostProcess.find_tips import find_tips
from collections import deque

def find_all_branch_paths(skel_img):

    branch_paths = []

    height, width = skel_img.shape

    skel_img_padded = np.pad(skel_img, pad_width=1)
    delta = [(-1, -1), (-1, 0), (-1, 1),
             (0, -1), (0, 1),
             (1, -1), (1, 0), (1, 1)]

    index = -1
    group = -1
    for i in range(2):
        point_dict = {}
        if i == 0:
            point_dict = find_branch_pts(skel_img)
        else:
            point_dict = find_tips(skel_img)
        for start_point, neighbors in point_dict.values():
            group += 1

            for point in neighbors:
                # The actual BFS algorithm
                #start_point = tip
                start_point_padded = [start_point[0], start_point[1]]
                if skel_img_padded[start_point[0]][start_point[1]] == 0 or skel_img_padded[point[0]][point[1]] == 0:
                    continue
                index += 1
                bfs = deque([point])
                path_points = np.array([start_point_padded, point])
                found = False

                prev_point = start_point_padded
                while len(bfs) > 0:
                    y, x = bfs.popleft()
                    # print(y,x)

                    # Look all 8 directions for a good path
                    hit_count = 0
                    next_yx = (x, y)
                    for dy, dx in delta:
                        yy, xx = y + dy, x + dx
                        # If the next position hasn't already been looked at and it's white
                        #if not (yy == prev_point[0] and xx == prev_point[1]) and ([yy, xx] not in neighbors) and skel_img_padded[yy][xx] > 0:
                        if not (yy == prev_point[0] and xx == prev_point[1]) and skel_img_padded[yy][xx] > 0:
                            hit_count += 1
                            next_yx = (yy, xx)
                    # 0 hit means line end without branch point
                    # 2 or more hit means line end with branch point
                    if hit_count == 1:
                        bfs.append(next_yx)
                        prev_point = path_points[-1]
                        skel_img_padded[prev_point[0]][prev_point[1]] = 0
                        path_points = np.append(path_points, [next_yx], axis=0)
                    elif hit_count == 0:
                        #prev_point = path_points[-1]
                        #skel_img_padded[prev_point[0]][prev_point[1]] = 0
                        #prev_point = path_points[-1]
                        #skel_img_padded[prev_point[0]][prev_point[1]] = 0
                        prev_point = path_points[-1]
                        skel_img_padded[prev_point[0]][prev_point[1]] = 0
                        break
                    elif hit_count >= 1 and (next_yx not in neighbors):
                        prev_point = path_points[-1]
                        skel_img_padded[prev_point[0]][prev_point[1]] = 0
                        #path_points = np.append(path_points, [next_yx], axis=0)
                        break

                path_points = path_points - 1
                branch_paths.append([index, group, path_points])

        skel_img_padded[start_point[0]][start_point[1]] = 0




    return branch_paths