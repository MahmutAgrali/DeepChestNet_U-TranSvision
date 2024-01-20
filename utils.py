# -*- coding: utf-8 -*-
"""
Created on Sun Nov 08 15:35:39 2021

@ This script contains same utils
"""

import numpy as np
import scipy.ndimage as ndimage
from skimage import measure,segmentation


def get_class_imbalance_ratio(img,is_integer=False):
    """
    @ This method provides to get class imbalance ratio
    @ params : 
        @@ img        : the image :: np.array
        @@ is_integer : the return type integer or float :: boolean
    @ returns : 
        @@ class_imbalance_ratio  : the class imbalance ratio :: integer or float
    """

    val, counts = np.unique(img,return_counts=True)      # get number of the pixel for each class 
    if is_integer:                                       # check return type                           
        return int((1 - (counts[0]/counts.sum()))*100)   #  return class imbalance ratio as integer
    else:
        return 1 - (counts[0]/counts.sum())              #  return class imbalance ratio as float
        



def divide_image(img,sub_img_shape=(64,64)):
    """
    @ This method divides the images for increase the class imbalance ratio
    @ Use for class imbalance problem 
    @ params : 
        @@ img            : the image                  :: np.array or nested list
        @@ sub_img_shape  : shape of the sub image     :: tuple
    @ returns : 
        @@ sub_imgs       : list of the sub image      :: list
    """
    if not isinstance(img, np.ndarray):   # check the image is np.array
        img   = np.array(img)             # convert the image to numpy array                                   
    img_shape = img.shape                 # get the image shape

    (n_h,n_w) = np.divide(img_shape,sub_img_shape)   # get number of the iteration for high and width of the image
    [n_h,n_w] = [int(n_h), int(n_w)]                 # convert high and width to integer   
    
    r_start  = 0                 # get row start        
    c_start  = 0                 # get column start    
    r_end    = sub_img_shape[0]  # get row end
    c_end    = sub_img_shape[1]  # get column end
    sub_imgs = list()            # get list for sub images

    for i in range(n_h):                                     # enumarate for high of the image
        for j in range(n_w):                                 # enumarate for width of the image
            temp_sub_img = img[r_start:r_end,c_start:c_end]  # get temp sub image 

            sub_imgs.append(temp_sub_img)                    # append temp sub image to the sub_imgs list

            c_start += sub_img_shape[1]                      # update c_start     
            c_end   += sub_img_shape[1]                      # update c_end
        
        r_start += sub_img_shape[0]                          # update r_start
        r_end   += sub_img_shape[0]                          # update r_end
        c_start = 0                                          # update c_start
        c_end   = sub_img_shape[1]                           # update c_end

    return sub_imgs                                          # return list of the sub images  

def join_image(sub_imgs,target_img_shape=(256,256)):
    """
    @ This method joins the sub images into an image 
    @ Use for class imbalance problem 
    @ params : 
        @@ sub_imgs          : the sub images               :: np.array or nested list
        @@ target_img_shape  : shape of the target image    :: tuple
    @ returns : 
        @@ img               : the joined image             :: np.array
    """
    if not isinstance(sub_imgs, np.ndarray): # check the image is np.array
        sub_imgs  = np.array(sub_imgs)       # convert the image to numpy array   
    (n,h,w)       = sub_imgs.shape           # get the image shape
    sub_img_shape = (h,w)                    # get shape of the sub images
    
    (n_h,n_w) = np.divide(target_img_shape,sub_img_shape) # get number of the iteration for high and width of the image
    [n_h,n_w] = [int(n_h), int(n_w)]                      # convert high and width to integer
    
    r_start = 0                       # get row start   
    c_start = 0                       # get column start    
    r_end   = sub_img_shape[0]        # get row end
    c_end   = sub_img_shape[1]        # get column end
    img = np.zeros(target_img_shape)  # get zeros for the joined image

    for i in range(n_h):                                              # enumarate for high of the joined image
        for j in range(n_w):                                          # enumarate for width of the joined image
            img[r_start:r_end,c_start:c_end] = sub_imgs[n_w*i+j,:,:]  # place part of the joined images


            c_start += sub_img_shape[1]                                # update c_start
            c_end   += sub_img_shape[1]                                # update c_end
        
        r_start += sub_img_shape[0]                          # update r_start
        r_end   += sub_img_shape[0]                          # update r_end
        c_start = 0                                          # update c_start
        c_end   = sub_img_shape[1]                           # update c_end

    return img                                               # return joined image


def compare_class_imbalance_ratio(img,sub_imgs):
    """
    @ This method compares class imbalance ratio
    @ params : 
        @@ img               : the image                                 :: np.array 
        @@ sub_imgs          : the sub images that are part of the image   :: tuple
    @ returns : 
        @@ img_ci_ratio      : class imbalance ratio for the image              :: float 
        @@ sub_imgs_ci_ratio : list of class imbalance ratio for the sub images :: list
    """  
    img_ci_ratio = get_class_imbalance_ratio(img)  # get class imbalance ratio for the image

    sub_imgs_ci_ratio = list()
    for sub_img in sub_imgs:
        tmp_sub_imgs_ci_ratio  = get_class_imbalance_ratio(sub_img)  # get class imbalance ratio for the sub images
        sub_imgs_ci_ratio.append(tmp_sub_imgs_ci_ratio)          # append class imbalance ratio to the list for the sub image     
    sub_imgs_ci_ratio          = np.array(sub_imgs_ci_ratio)     # convert list to np.array 
    return img_ci_ratio, sub_imgs_ci_ratio                       # return class imbalance ratio



def generate_markers(image):
    """
    Generates markers for a given image.
    
    Parameters: image
    
    Returns: Internal Marker, External Marker, Watershed Marker
    """
    
    #Creation of the internal Marker
    marker_internal = image < 180
    marker_internal = segmentation.clear_border(marker_internal)
    marker_internal_labels = measure.label(marker_internal)
    
    areas = [r.area for r in measure.regionprops(marker_internal_labels)]
    areas.sort()
    
    if len(areas) > 2:
        for region in measure.regionprops(marker_internal_labels):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                        marker_internal_labels[coordinates[0], coordinates[1]] = 0
    
    marker_internal = marker_internal_labels > 0
    
    # Creation of the External Marker
    external_a = ndimage.binary_dilation(marker_internal, iterations=10)
    external_b = ndimage.binary_dilation(marker_internal, iterations=55)
    marker_external = external_b ^ external_a
    
    # Creation of the Watershed Marker
    marker_watershed = np.zeros((512, 512), dtype=np.int)
    marker_watershed += marker_internal * 255
    marker_watershed += marker_external * 128
    
    return marker_internal, marker_external, marker_watershed

def seperate_lungs(image, iterations = 1):
    """
    Segments lungs using various techniques.
    
    Parameters: image (Scan image), iterations (more iterations, more accurate mask)
    
    Returns: 
        - Segmented Lung
        - Lung Filter
        - Outline Lung
        - Watershed Lung
        - Sobel Gradient
    """
    
    marker_internal, marker_external, marker_watershed = generate_markers(image)
    
    
    '''
    Creation of Sobel Gradient
    '''
    
    # Sobel-Gradient
    sobel_filtered_dx = ndimage.sobel(image, 1)
    sobel_filtered_dy = ndimage.sobel(image, 0)
    sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
    sobel_gradient *= 255.0 / np.max(sobel_gradient)
    
    
    '''
    Using the watershed algorithm
    
    
    We pass the image convoluted by sobel operator and the watershed marker
    to morphology.watershed and get a matrix matrix labeled using the 
    watershed segmentation algorithm.
    '''
    watershed = segmentation.watershed(sobel_gradient, marker_watershed)
    
    '''
    Reducing the image to outlines after Watershed algorithm
    '''
    outline = ndimage.morphological_gradient(watershed, size=(3,3))
    outline = outline.astype(bool)
    
    
    '''
    Black Top-hat Morphology:
    
    The black top hat of an image is defined as its morphological closing
    minus the original image. This operation returns the dark spots of the
    image that are smaller than the structuring element. Note that dark 
    spots in the original image are bright spots after the black top hat.
    '''
    
    # Structuring element used for the filter
    blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],
                        [0, 1, 1, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 1, 1, 0],
                        [0, 0, 1, 1, 1, 0, 0]]
    
    blackhat_struct = ndimage.iterate_structure(blackhat_struct, iterations)
    
    # Perform Black Top-hat filter
    outline += ndimage.black_tophat(outline, structure=blackhat_struct)
    
    '''
    Generate lung filter using internal marker and outline.
    '''
    lungfilter = np.bitwise_or(marker_internal, outline)
    lungfilter = ndimage.morphology.binary_closing(lungfilter, structure=np.ones((5,5)), iterations=3)
    
    '''
    Segment lung using lungfilter and the image.
    '''
    segmented = np.where(lungfilter == 1, image, 0*np.ones((512, 512)))
    
    return segmented, lungfilter, outline, watershed, sobel_gradient

def iou_Score(y_true, y_pred):
    intersection = np.sum(np.multiply(y_true, y_pred))
    union = np.sum(y_true)+np.sum(y_pred) - intersection
    x = intersection = (intersection + 1e-15)/(union + 1e-15)
    return x

def dicom_to_JPG(dcm_img):
    rows = 512
    cols = 512
    
    window_center = dcm_img.WindowCenter
    window_width  = dcm_img.WindowWidth
    rescale_intercept = dcm_img.RescaleIntercept
    rescale_slope     = dcm_img.RescaleSlope
    
    window_max        = int(window_center+window_width/2)
    window_min        = int(window_center-window_width/2)
   
    
   
    
    new_img = np.zeros((rows,cols))
    
    dcm_img = dcm_img.pixel_array
    img = dcm_img * rescale_slope + rescale_intercept
    new_img =  np.divide(np.subtract(img,window_min),window_max-window_min)*255
    new_img[dcm_img > window_max] = 255
    new_img[dcm_img < window_min] = 0   
  
    return new_img

