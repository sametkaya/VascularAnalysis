import glob
import os

import numpy as np
import SimpleITK as sitk


input_folder = r"C:\Users\skaya\PycharmProjects\VascularAnalysis\ImageProcessing\stack\stack full\z"
output_folder = r"C:\Users\skaya\PycharmProjects\VascularAnalysis\ImageProcessing\stack\stack full\c"

# Get list of TIFF files in folder
tiff_filenames = [os.path.join(input_folder, filename) for filename in os.listdir(input_folder) if filename.endswith(".tiff")]

# Read in TIFF images as a list of SimpleITK objects
tiff_images = [sitk.ReadImage(filename) for filename in tiff_filenames]

# Convert list of 2D images to 3D image
image3d = sitk.JoinSeries(tiff_images)

# Set image spacing
spacing = tiff_images[0].GetSpacing() + (0.1,)  # add spacing for Z-axis
image3d.SetSpacing(spacing)

# Convert to NIfTI and save
sitk.WriteImage(image3d, "output1.nii.gz")