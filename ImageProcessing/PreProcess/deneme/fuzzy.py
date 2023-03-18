import numpy as np
import skfuzzy as fuzz
import os
import tifffile as tiff

# Define path to folder containing TIFF stacks
folder_path = r"C:\Users\skaya\PycharmProjects\VascularAnalysis\ImageProcessing\stack\stack full\z"

# Get list of TIFF files in folder
files = os.listdir(folder_path)

# Loop over TIFF files and convert them to numpy matrices
matrices = []
for file in files:
    if file.endswith('.tiff'):
        # Load TIFF image stack
        stack = tiff.imread(os.path.join(folder_path, file))

        # Convert stack to numpy matrix
        matrix = np.array(stack)

        # Append matrix to list of matrices
        matrices.append(matrix)

# Stack matrices along a new axis to create a 4D numpy array
pixel_matrix = np.stack(matrices, axis=0)

# Generate random 3D pixel matrix
#pixel_matrix = np.random.rand(10, 10, 10)

# Define fuzziness parameter
m = 2.0

# Define maximum number of iterations
max_iter = 100

# Define tolerance for automatically determining number of clusters
tolerance = 0.01

# Define pixel distance function
def dist(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

# Flatten the pixel matrix to perform clustering
pixel_data = np.reshape(pixel_matrix, (-1, pixel_matrix.shape[-1]))

# Compute pairwise distances between pixels
distances = np.zeros((pixel_data.shape[0], pixel_data.shape[0]))
for i in range(pixel_data.shape[0]):
    for j in range(i+1, pixel_data.shape[0]):
        d = dist(pixel_data[i], pixel_data[j])
        distances[i,j] = d
        distances[j,i] = d

# Perform fuzzy clustering and automatically determine number of clusters
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    pixel_data.T, c=2, m=m, error=0.005, maxiter=max_iter, init=None, dist=distances)

# Reshape the cluster membership matrix
n_clusters = u.shape[0]
u = u.reshape(n_clusters, -1)

# Determine the most likely cluster for each pixel
labels = np.argmax(u, axis=0)

# Reshape the labels into a 3D array
labels = labels.reshape(pixel_matrix.shape)

# Print the resulting labels and number of clusters
print("Number of clusters:", n_clusters)
print("Labels:")
print(labels)