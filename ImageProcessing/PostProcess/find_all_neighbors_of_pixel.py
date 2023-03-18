
def find_all_neighbors_of_pixel(skel_image, point):
    # All 8 directions
    delta = [(-1, -1), (-1, 0), (-1, 1),
             (0, -1),           (0, 1),
             (1, -1),   (1, 0), (1, 1)]