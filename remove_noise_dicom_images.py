# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 08:03:21 2021

@ This script contains the functions that are removed the noise from a Dicom image
"""

#####################################################
# @ import required modules
#####################################################
import numpy as np
from pydicom import dcmread
import pydicom
import config
import glob
from skimage import morphology
from scipy import ndimage 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def transform_to_hu(medical_image, image):
    """
    @ This method transfroms from intensity value to hu value
    @ params : 
        @@ medical_image : the medical image object :: 
        @@ image    : the dataframe :: pandas.Dataframe
    @ returns : 
        @@ 
    """
    
    intercept = medical_image.RescaleIntercept   # get intercept
    slope     = medical_image.RescaleSlope       # get slope  
    hu_image  = image * slope + intercept        # get hu image

    return hu_image                              # return hu image

def window_image(image, window_center, window_width):
    """
    @ This method transfroms from intensity value to hu value
    @ params : 
        @@ medical_image : the medical image object :: 
        @@ image    : the dataframe :: pandas.Dataframe
    @ returns : 
        @@ 
    """
    img_min                              = window_center - window_width // 2  # get image minumum
    img_max                              = window_center + window_width // 2  # get image maximum
    window_image                         = image.copy()                       # copy image
    window_image[window_image < img_min] = img_min                            # set limit
    window_image[window_image > img_max] = img_max                            # set limit  
    
    return window_image                                                       # return image
    
def remove_noise(img=None,file_path=None, display=False):
    """
    @ This method remove noise
    @ params : 
        @@ img           : the image     :: np.array 
        @@ file_path     : the file path :: String
        @@ display       : the status for display : boolean
    @ returns : 
        @@ masked_image  : image :: np.array
    """
    
    if file_path != None:                             # check file path 
        medical_image = pydicom.read_file(file_path)  # get medical image
        image         = medical_image.pixel_array     # get pixel array of medical image
    else:
        assert type(img) != None             # raise img is None
        medical_image     = img              # get medical image   
        image             = img.pixel_array  # get pixel array of medical image
    
    
    hu_image    = transform_to_hu(medical_image, image)                                         # get hu image
    lung_image = window_image(hu_image, medical_image.WindowCenter, medical_image.WindowWidth) #bone windowing
    
    segmentation     = morphology.dilation(lung_image, np.ones((1, 1)))  # apply morphological dilation process
    labels, label_nb = ndimage.label(segmentation)                       # get label 
    
    label_count    = np.bincount(labels.ravel().astype(np.int))          # get label count
    label_count[0] = 0                                                   # set label count

    mask = labels == label_count.argmax()                                # get argmax for label count     
 
    mask         = morphology.dilation(mask, np.ones((1, 1)))            # apply morphological dilation process
    mask         = ndimage.morphology.binary_fill_holes(mask)            # apply morphological binary fill process
    mask         = morphology.dilation(mask, np.ones((3, 3)))            # apply morphological dilation process
    masked_image = mask * lung_image                                     # get masked image     
    return masked_image                                                  # return masked image

if __name__ == "__main__":
    ############################################
    # Test the script
    ############################################

    # Define dataset paths and extensions
    DATASET                     = config.DATASET                       # get dataset path from config
    DATASET_ORIGINAL_DICOM_PATH = config.DATASET_ORIGINAL_DICOM_PATH   # get dataset orginal dicom path from config
    DATASET_ORIGINAL_JPG_PATH   = config.DATASET_ORIGINAL_JPG_PATH     # get dataset original jpg path from config
    DATASET_MASK_PATH           = config.DATASET_MASK_PATH             # get dataset mask path from config
    DATASET_ORIGINAL_DICOM_EXT  = config.DATASET_ORIGINAL_DICOM_EXT    # get dataset dicom extension from config
    DATASET_ORIGINAL_JPG_EXT    = config.DATASET_ORIGINAL_JPG_EXT      # get dataset jpg extension from config
    DATASET_MASK_EXT            = config.DATASET_MASK_EXT              # get dataset mask png extension from config
    IMAGE_DIR_NPY               = config.IMAGE_DIR_NPY                 # get original image path of file that has npy extension
    MASKS_DIR_NPY               = config.MASKS_DIR_NPY                 # get mask image path of file that has npy extension

    original_dicom_list = sorted(glob.glob(DATASET_ORIGINAL_DICOM_PATH+'*'+DATASET_ORIGINAL_DICOM_EXT))  # get original dicom list


    masked_image = remove_noise(original_dicom_list[np.random.randint(1,11800)])  # load random Dicom image and remove whose noise 
   
    plt.figure()                          # generate a new figure
    plt.imshow(masked_image,cmap='gray')  # show masked image
    
    scaler = MinMaxScaler()                                                                                         # generate min-max scaler
    masked_image=scaler.fit_transform(masked_image.reshape(-1, masked_image.shape[-1])).reshape(masked_image.shape) # apply min-max scaler to image

    plt.figure()                          # generate a new figure
    plt.imshow(masked_image,cmap='gray')  # show masked image
    