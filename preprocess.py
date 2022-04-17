import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


def get_data(i): 

    BASE_FILE_PATH = "./archive/"

    SEGMENTATION_PATH = BASE_FILE_PATH+'segmentations/'

    img_slices = []
    for i in range(0,5): #change this value 
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


        #Slice images
        slc_nonzero_val = {}
        z_len = image.shape[2]
        for j in range(z_len):
            new_img_slice = image[:,:,j]
            if not np.all(new_img_slice < 0): 
                slc_nonzero_val[i] = j
                img_slices.append(new_img_slice) #adds to new_img_slice only if there is some nonnegative value found 
            
            # **can call save_vals function here**


    return img_slices

#Compression of images 3D --> 2D through reshape
"""
    new_arr = image.reshape(-1, image.shape[-1])
    print(np.shape(new_arr))

    mask = np.reshape(mask, (-1,512,512))
    print(np.shape(new_arr))
"""

def save_vals(new_img_slice, i, j): 
    """
    Saves values for recorded and unrecorded data in a file according to whether there is some positive value found in the image which would 
    represent a tumor
    Inputs: i - Val of image we are on; j - z-value on axis 
    Returns: None 
    """
    if not np.all(new_img_slice < 0): 
        f = open("recorded.txt", "a")
        f.write(f"{i},  {j} \n")
        f.close() 
    else: 
        f = open("ignored.txt", "a")
        f.write(f"{i}, {j} \n")
        f.close()

def printNewImgVals(new_img_slice): 
    for i in range(len(new_img_slice)): 
        for j in range(len(new_img_slice[0])):
            print(new_img_slice[i,j])
    