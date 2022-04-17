import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
epi_img = nib.load('/Users/abigailmarks/Desktop/brown/csci1470/fp/volume-0.nii')
epi_img_data = epi_img.get_fdata()
print(epi_img_data.shape)
epi_img2 = nib.load('/Users/abigailmarks/Desktop/brown/csci1470/fp/segmentation-0.nii')
epi_img_data2 = epi_img2.get_fdata()
print(epi_img_data2.shape)

def show_slices(slices):
   """ Function to display row of image slices """
   fig, axes = plt.subplots(1, len(slices))
   for i, slice in enumerate(slices):
       axes[i].imshow(slice.T, cmap="gray", origin="lower")


nonzero =np.nonzero(epi_img_data2)[2]
print(np.nonzero(epi_img_data2))

for i in np.unique(nonzero):
    plt.imshow(epi_img_data2[:,:,i])
    plt.show()
    plt.imshow(epi_img_data[:,:,i],cmap="gray")
    plt.show()