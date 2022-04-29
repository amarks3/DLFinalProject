import os
from tkinter import Y
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import h5py


def get_data(i_val): 
    """
    Returns the images and labels as a tuple 
    """

    BASE_FILE_PATH = "./archive/"

    SEGMENTATION_PATH = BASE_FILE_PATH+'segmentations/'

    img_slices = []
    labels = []
    for i in range(0,i_val): #change this value 
        print("i: ", i )
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
        #print("lengths ", z_len, mask.shape[2])
        for j in range(z_len):
            new_img_slice = image[:,:,j]
            new_label_slice = mask[:,:,j]
            if not np.all(new_label_slice == 0): 
                slc_nonzero_val[i] = j
                img_slices.append(new_img_slice) #adds to new_img_slice only if there is some nonnegative value found 
                labels.append(new_label_slice) #adds labels if adding new image 
                # **can call save_vals function here**
                save_vals(new_img_slice, new_label_slice)
            


    return img_slices, labels

#Compression of images 3D --> 2D through reshape
"""
    new_arr = image.reshape(-1, image.shape[-1])
    print(np.shape(new_arr))
    mask = np.reshape(mask, (-1,512,512))
    print(np.shape(new_arr))
"""

def save_vals(X_train_data, Y_train_labels): 
    """
    Saves values for recorded and unrecorded data in a file according to whether there is some positive value found in the image which would 
    represent a tumor
    Inputs: i - Val of image we are on; j - z-value on axis 
    Returns: None 
    """
    with h5py.File('.\PreprocessedData.h5', 'w') as hf:
        hf.create_dataset("X_train", data=X_train_data)
        hf.create_dataset("Y_train", data=Y_train_labels)

    """f = open("recorded.txt", "a")
    f.write(f"{i},  {j} \n")
    f.close() """
    """f = open("ignored.txt", "a")
    f.write(f"{i}, {j} \n")
    f.close()"""

def printNewImgVals(new_img_slice): 
    for i in range(len(new_img_slice)): 
        for j in range(len(new_img_slice[0])):
            print(new_img_slice[i,j])