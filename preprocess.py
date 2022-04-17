import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


BASE_FILE_PATH = "/Users/abigailmarks/Downloads/archive/"

SEGMENTATION_PATH = BASE_FILE_PATH+'segmentations/'

for i in range(0,51):
    segmentation_file = SEGMENTATION_PATH + 'segmentation-'+str(i)+'.nii'
    folder_num=1
    if i <11:
        folder_num=1
    elif i <21:
        folder_num=2
    elif i<31:
        folder_num=3
    elif i<41:
        folder_num=4
    else:
        folder_num=5

    image_file = f'{BASE_FILE_PATH}volume_pt{folder_num}/volume-{i}.nii'
    mask0 = nib.load(segmentation_file)
    mask = mask0.get_fdata()
    image0 = nib.load(image_file)
    image = image0.get_fdata()

    new_arr = image.reshape(-1, image.shape[-1])
    print(np.shape(ims))

    mask = np.reshape(mask, (-1,512,512))
    print(np.shape(ims))