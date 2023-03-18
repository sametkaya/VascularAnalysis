import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

mat=np.random.randint(0, 255, size=(10,10,10))
#
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

arr = np.array([[[1, 2, 3], [4, 5, 6] ,[7, 8, 9]],
                [[10, 11, 12],[13, 14, 15], [16, 17, 18]],
                [[19, 20, 21], [22, 23, 24], [25, 26, 27]]])
for x in range(3):
  for y in range(3):
    for z in range(3):
      print(x,y,z,mat[x][y][z])
      ax.scatter(x, y, z, c="rgb("+str(mat[x][y][z])+","+str(mat[x][y][z])+","+str(mat[x][y][z])+")")

plt.show()

# for data, color, group in zip(data, colors, groups):
#     x, y, z = data
#     ax.scatter(x, y, z, alpha=0.8, edgecolors='none', s=30)
i=0