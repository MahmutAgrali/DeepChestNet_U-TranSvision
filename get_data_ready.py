# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:48:52 2021

@ Use this code to get Covid19 dataset ready for semantic segmentation. 
@ Code can be divided into a few parts....

"""

# Import required modules
import numpy as np
import glob
from sklearn.preprocessing import MinMaxScaler
import cv2
from skimage.transform import resize
from skimage.io import imread
import config
import os
from remove_noise_dicom_images import remove_noise
import pandas as pd
from shutil import copy

def ensure_dir(file_path):
    """
    @ This method provides to check the file path exists
    @ params : 
        @@ file_path: the file path :: string
    @ returns : 
        @@
    """
    directory  = os.path.dirname(file_path)    # get directory name
    
    if not os.path.exists(directory):          # check the file exists
        os.makedirs(directory)                 # generate a new directory


###########################################
#PART 1: Load, crop and resize sample images
#Includes, dividing each image by its max to scale them to [0,1]
###########################################

# Define dataset paths and extensions
DATASET                     = config.DATASET                       # get dataset path from config
DATASET_ORIGINAL_DICOM_PATH = config.DATASET_ORIGINAL_DICOM_PATH   # get dataset orginal dicom path from config
DATASET_ORIGINAL_JPG_PATH   = config.DATASET_ORIGINAL_JPG_PATH     # get dataset original jpg path from config
DATASET_COV_MASK_PATH           = config.DATASET_COV_MASK_PATH             # get dataset mask path from config
DATASET_ORIGINAL_DICOM_EXT  = config.DATASET_ORIGINAL_DICOM_EXT    # get dataset dicom extension from config
DATASET_ORIGINAL_JPG_EXT    = config.DATASET_ORIGINAL_JPG_EXT      # get dataset jpg extension from config
DATASET_MASK_EXT            = config.DATASET_MASK_EXT              # get dataset mask png extension from config
IMAGE_DIR_NPY               = config.IMAGE_DIR_NPY                 # get original image path of file that has npy extension
MASKS_DIR_NPY               = config.MASKS_DIR_NPY                 # get mask image path of file that has npy extension
results_dir                 = config.results_dir                   # get results directory

# Make directory
ensure_dir(config.DATASET)                       # check the directy exists
ensure_dir(config.INPUT_DIR_NPY)                 # check the directy exists
ensure_dir(config.IMAGE_DIR_NPY)                 # check the directy exists
ensure_dir(config.MASKS_DIR_NPY)                 # check the directy exists
ensure_dir(config.DATASET_SPLIT)                 # check the directy exists
ensure_dir(config.DATASET+results_dir)           # check the directy exists

df = pd.DataFrame(columns=["Epoch", 
                            "Batch",
                            "Architecture",
                            "Optimizer",
                            "Learning_Rate",
                            "Momentum",
                            "Kernel_Initializer",
                            "Seed_Number",                        
                            "Max_IOU_Score",                     
                            "Max_Val_IOU_Score",
                            "Dataset",
                            "Date",
                            "Annotions"])                             # generate dataframe using the parameters
if(not os.path.isfile(DATASET+results_dir+"all_results.xlsx")):
    df.to_excel(DATASET+results_dir+"all_results.xlsx",index=False)                  # save the df to the xlsx file

# Sort images lists 
original_dicom_image_paths = sorted(glob.glob(DATASET_ORIGINAL_DICOM_PATH+'*'+DATASET_ORIGINAL_DICOM_EXT)) # get list of the original dicom image paths
original_jpg_image_paths = sorted(glob.glob(DATASET_ORIGINAL_JPG_PATH+'*'+DATASET_ORIGINAL_JPG_EXT))     # get list of the original jpg image paths
mask_image_paths = sorted(glob.glob(config.DATASET_LUNG_MASK_PATH+'/*/*'+DATASET_MASK_EXT))                               # get list of the mask image paths



scaler = MinMaxScaler() # generate min-max scaler


for i,img_path in enumerate(mask_image_paths):   # enumarate list of the original dicom image paths 
    _,file_name    = os.path.split(img_path)            # get file name 
    file_name      = file_name.replace(".png","")       # replace dcm extension
        
    original_dicom = glob.glob(DATASET_ORIGINAL_DICOM_PATH+file_name+".dcm")
    if len(original_dicom) != 1:
        raise "There is no original dicom image."
    # read images and masks
    temp_image_original_dicom       = remove_noise(file_path=original_dicom[0], display=False)     # read the image from path and remove nose in the image
    temp_mask                       = imread(img_path,cv2.IMREAD_GRAYSCALE)    # load for png extension
    
    temp_mask    = temp_mask.astype(np.bool_)   # convert mask to boolean
    
    temp_image_original_dicom           = scaler.fit_transform(temp_image_original_dicom.reshape(-1, temp_image_original_dicom.shape[-1])).reshape(temp_image_original_dicom.shape) # apply min-max scaler to image

    temp_image_original_dicom   = resize(temp_image_original_dicom, (256, 256), mode='constant', preserve_range=True)    # Resize images to 256x256 
    temp_mask                   = resize(temp_mask, (256, 256), mode='constant', preserve_range=True)                    # Resize images to 256x256
    
   
    np.save(IMAGE_DIR_NPY+'image_'+str(file_name)+'.npy', temp_image_original_dicom)    # Save image to npy
    np.save(MASKS_DIR_NPY+'mask_'+str(file_name)+'.npy', temp_mask)                     # Save image to npy
        
################################################################
# Part 2 : Train, validation and test split
# Split training data into train and validation, and split test data

"""
Code for splitting folder into train, test, and val.
Once the new folders are created rename them and arrange in the format below to be used
for semantic segmentation using data generators. 

pip install split-folders
"""
import splitfolders  # or import split_folders


input_folder    = config.INPUT_DIR_NPY   # get input path 
output_folder   = config.DATASET_SPLIT   # get output path

splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.68,.17, .15), group_prefix=None)                                    # split the dataset as train, val and test


################################################################

