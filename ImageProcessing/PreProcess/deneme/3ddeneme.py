import os

import numpy as np
import SimpleITK as sitk

import os
import numpy as np
import SimpleITK as sitk


def otsu_threshold_3d_folder(input_folder, output_folder):
    # Make sure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop over all files in the input folder
    for filename in os.listdir(input_folder):
        # Load the image file
        filepath = os.path.join(input_folder, filename)
        sitk_img = sitk.ReadImage(filepath)

        # Compute Otsu threshold for each slice
        otsu_filter = sitk.OtsuThresholdImageFilter()
        otsu_filter.SetInsideValue(0)
        otsu_filter.SetOutsideValue(1)

        thresholded_slices = []
        for z in range(sitk_img.GetSize()[2]):
            slice_img = sitk_img[:, :, z]
            threshold = otsu_filter.Execute(slice_img)
            thresholded_slices.append(sitk.GetArrayFromImage(threshold))

        # Merge thresholded slices into a 3D array
        thresholded_stack = np.stack(thresholded_slices, axis=2)

        # Save the thresholded stack as a new image file
        output_filepath = os.path.join(output_folder, filename)
        sitk.WriteImage(sitk.GetImageFromArray(thresholded_stack), output_filepath)




input_folder = r"C:\Users\skaya\PycharmProjects\VascularAnalysis\ImageProcessing\stack\stack full\z"
output_folder = r"C:\Users\skaya\PycharmProjects\VascularAnalysis\ImageProcessing\stack\stack full\c"

otsu_threshold_3d_folder(input_folder, output_folder)