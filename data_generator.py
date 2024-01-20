# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 13:04:52 2021

@ Custom data generator to work with Covid19 dataset.
@ Can be used as a template to create your own custom data generators. 
@ No image processing operations are performed here, just load data from local directory in batches. 
"""

import os
import numpy as np
import pandas as pd
from skimage.transform import resize
from utils import  divide_image,join_image,compare_class_imbalance_ratio,get_class_imbalance_ratio
from skimage import exposure,filters,segmentation,morphology


def load_image(img_dir, img_list,is_expand_dims=False,image_processing=False):
    """
    @ This method loads list of images or masks
    @ params : 
        @@ img_dir   : image directory :: string
        @@ img_list  : list of image :: list
    @ returns : 
        @@ images : list of the image :: np.array
    """
    
    # Find npy file and get image from the file 
    images     = []                               # initialize the list to append the images
    for i, image_name in enumerate(img_list):     # enumarate image list
        if (image_name.split('.')[-1] == 'npy'):   # check file extension

            image = np.load(img_dir+image_name)   # load the image from .npy file
            # cv2.imshow("image1",image) # show original image

            # image  = resize(image, (128, 128), mode='constant', preserve_range=True)    # Resize images to 256x256#
            if image_processing:
                # clahe = cv2.createCLAHE(clipLimit=0.45, tileGridSize=(8, 8))
                # image = clahe.apply(image)
                # image = filters.unsharp_mask(image,radius=5)
                # image = exposure.equalize_adapthist(image)#,clip_limit=0.1)
                # image[image == 0] = 0.001
                pass
                # val = filters.threshold_otsu(image)
                # temp_image = np.copy(image)
                # temp_image[image>=val] = 1
                # temp_image[image<val] = 0
                # temp_image = cv2.dilate(temp_image, morphology.disk(3))
                # image = (temp_image)*image
            if is_expand_dims:
                image = np.expand_dims(image,axis=-1)#np.resize(image,(256,256))
            # image = np.resize(image,(256,256))
            # cv2.imshow("image",image) # show original image
            # cv2.waitKey(0)
            images.append(image)                  # append the image to the list
    
    images = np.array(images)                     # convert the list of images from list to np.array 
    
    return(images)                                # return list of image

def load_image2(img_dir, img_list,is_expand_dims=False,image_processing=False,k_fold=False,img_dir2=None):
    """
    @ This method loads list of images or masks
    @ params : 
        @@ img_dir   : image directory :: string
        @@ img_list  : list of image :: list
    @ returns : 
        @@ images : list of the image :: np.array
    """
    
    # Find npy file and get image from the file 
    images     = []                               # initialize the list to append the images
    for i, image_name in enumerate(img_list):     # enumarate image list
        if (image_name.split('.')[-1] == 'npy'):   # check file extension

            image = np.load(img_dir+image_name)   # load the image from .npy file
            # cv2.imshow("image1",image) # show original image
            if is_expand_dims:
                image = np.expand_dims(image,axis=-1)#np.resize(image,(256,256))
            # image = np.resize(image,(256,256))
            # cv2.imshow("image",image) # show original image
            # cv2.waitKey(0)
            images.append(image)                  # append the image to the list
    
            if k_fold:
                image = np.load(img_dir2+image_name.replace("orginal","generated"))   # load the image from .npy file
                # cv2.imshow("image1",image) # show original image
                if is_expand_dims:
                    image = np.expand_dims(image,axis=-1)#np.resize(image,(256,256))
                # image = np.resize(image,(256,256))
                # cv2.imshow("image",image) # show original image
                # cv2.waitKey(0)
                images.append(image)                  # append the image to the list

    images = np.array(images)                     # convert the list of images from list to np.array 
    
    return(images)                                # return list of image


def load_images(img_dir, img_list,mask_dir,mask_list,is_divide=True):
    """
    @ This method loads list of images or masks
    @ params : 
        @@ img_dir   : image directory :: string
        @@ img_list  : list of image :: list
        @@ is_divide : if it is True, the image is divided : boolean
    @ returns : 
        @@ tuple images : tuple from list of the image :: tuple
    """
    
    # Find npy file and get image from the file 
    mask_list_1,mask_list_2,mask_list_3,mask_list_4,mask_list_5,mask_list_6,mask_list_7,mask_list_8,mask_list_9,mask_list_10,mask_list_11,mask_list_12,mask_list_13,mask_list_14,mask_list_15,mask_list_16 = list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list()
    image_list_1,image_list_2,image_list_3,image_list_4,image_list_5,image_list_6,image_list_7,image_list_8,image_list_9,image_list_10,image_list_11,image_list_12,image_list_13,image_list_14,image_list_15,image_list_16 = list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list()

    images     = []                               # initialize the list to append the images
    masks     = []                                # initialize the list to append the mask images
    for i, mask_name in enumerate(mask_list):     # enumarate mask image list
        if (mask_name.split('.')[-1] == 'npy'):   # check file extension
            
            mask            = np.load(mask_dir+mask_name)               # load the mask image from .npy file
            image           = np.load(img_dir+img_list[i])                 # load the image from .npy file
             
            sub_masks       = divide_image(mask,sub_img_shape=(64,64))  # divide the mask image
            sub_images      = divide_image(image,sub_img_shape=(64,64)) # divide the image
            
            temp_images    = []                               # initialize the list to append the images
            temp_masks     = []                                # initialize the list to append the mask images
            for j, sub_mask in enumerate(sub_masks):             # enumarate the sub mask images
                if get_class_imbalance_ratio(sub_mask) >= 0.0:   # check class imbalance ratio for the sub mask image to eliminate the image less than %5
                    temp_masks.append(np.expand_dims(sub_mask,axis=-1))                       # append the mask image to the list          
                    temp_images.append(np.expand_dims(sub_images[j],axis=-1))                 # append the image to the list       

            mask_list_1.append(temp_masks[0])
            mask_list_2.append(temp_masks[1])
            mask_list_3.append(temp_masks[2])
            mask_list_4.append(temp_masks[3])
            mask_list_5.append(temp_masks[4])
            mask_list_6.append(temp_masks[5])
            mask_list_7.append(temp_masks[6])
            mask_list_8.append(temp_masks[7])
            mask_list_9.append(temp_masks[8])
            mask_list_10.append(temp_masks[9])
            mask_list_11.append(temp_masks[10])
            mask_list_12.append(temp_masks[11])
            mask_list_13.append(temp_masks[12])
            mask_list_14.append(temp_masks[13])
            mask_list_15.append(temp_masks[14])
            mask_list_16.append(temp_masks[15])
            
            image_list_1.append(temp_images[0])
            image_list_2.append(temp_images[1])
            image_list_3.append(temp_images[2])
            image_list_4.append(temp_images[3])
            image_list_5.append(temp_images[4])
            image_list_6.append(temp_images[5])
            image_list_7.append(temp_images[6])
            image_list_8.append(temp_images[7])
            image_list_9.append(temp_images[8])
            image_list_10.append(temp_images[9])
            image_list_11.append(temp_images[10])
            image_list_12.append(temp_images[11])
            image_list_13.append(temp_images[12])
            image_list_14.append(temp_images[13])
            image_list_15.append(temp_images[14])
            image_list_16.append(temp_images[15])

    masks  = np.array(mask_list_1),np.array(mask_list_2),np.array(mask_list_3),np.array(mask_list_4),np.array(mask_list_5),np.array(mask_list_6),np.array(mask_list_7),np.array(mask_list_8),np.array(mask_list_9),np.array(mask_list_10),np.array(mask_list_11),np.array(mask_list_12),np.array(mask_list_13),np.array(mask_list_14),np.array(mask_list_15),np.array(mask_list_16)                      # convert the list of images from list to np.array 
    images = np.array(image_list_1),np.array(image_list_2),np.array(image_list_3),np.array(image_list_4),np.array(image_list_5),np.array(image_list_6),np.array(image_list_7),np.array(image_list_8),np.array(image_list_9),np.array(image_list_10),np.array(image_list_11),np.array(image_list_12),np.array(image_list_13),np.array(image_list_14),np.array(image_list_15),np.array(image_list_16)                     # convert the list of images from list to np.array 

    return(images,masks)                                # return list of image



def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size,is_divide=False,k_fold=False,img_dir2=""):
    """
    @ This method loads the images and masks by batch 
    @ Keras datagenerator does not support 3d
    @ params : 
        @@ img_dir    : original images path               :: string
        @@ img_list   : list of the original images path   :: list
        @@ mask_dir   : mask images path                   :: string
        @@ mask_list  : list of the mask images path       :: list
        @@ batch_size : batch size of the dataset          :: integer
        @@ is_divide : if it is True, the image is divided :: boolean

    @ returns : 
        @@ images : list of the image :: np.array
    """
    
    L = len(img_list)  # get length of the original image list 
    
    while True:                     #keras needs the generator infinite, so we will use while true  

        batch_start  = 0            # initialize batch start variable
        batch_end    = batch_size   # initiazlize batch end variable

        while batch_start < L:        
            limit  = min(batch_end, L)                                  # get batch limit         
            if is_divide:                                                   # check the variable to divide the image
                X,Y    = load_images(img_dir, img_list[batch_start:limit],mask_dir, mask_list[batch_start:limit])     # load masks and images 
            else:
                img_dir2_img = img_dir2 + "images/"
                img_dir2_mask = img_dir2 + "masks/"

                X      = load_image2(img_dir, img_list[batch_start:limit],k_fold=k_fold,img_dir2=img_dir2_img)     # load original images 
                Y      = load_image2(mask_dir, mask_list[batch_start:limit],k_fold=k_fold,img_dir2=img_dir2_mask)   # load mask images

            yield (X,Y)                                                 # return a tuple that in two numpy arrays with batch_size samples     

            batch_start += batch_size                                   # increase batch_start variable by batch_size
            batch_end += batch_size                                     # increase batch_end variable by batch_size

def imageLoaderSpervision(img_dir, img_list, mask_dir, mask_list, batch_size):
    """
    @ This method loads the images and masks by batch 
    @ Keras datagenerator does not support 3d
    @ params : 
        @@ img_dir    : original images path               :: string
        @@ img_list   : list of the original images path   :: list
        @@ mask_dir   : mask images path                   :: string
        @@ mask_list  : list of the mask images path       :: list
        @@ batch_size : batch size of the dataset          :: integer
        @@ is_divide : if it is True, the image is divided :: boolean

    @ returns : 
        @@ images : list of the image :: np.array
    """
    
    L = len(img_list)  # get length of the original image list 

    while True:                     #keras needs the generator infinite, so we will use while true  

        batch_start  = 0            # initialize batch start variable
        batch_end    = batch_size   # initiazlize batch end variable

        while batch_start < L:        
            limit  = min(batch_end, L)                                  # get batch limit         

            X      = load_image(img_dir, img_list[batch_start:limit],image_processing=True)     # load original images 
            Y      = load_image(mask_dir, mask_list[batch_start:limit])   # load mask images


            yield (X,Y)                                                 # return a tuple that in two numpy arrays with batch_size samples     

            batch_start += batch_size                                   # increase batch_start variable by batch_size
            batch_end += batch_size                                     # increase batch_end variable by batch_size

if __name__ == "__main__":
    ############################################
    # Test the generator
    ############################################

    from matplotlib import pyplot as plt
    import random
    
    DATASET                   = "DATASETS/DATASET_DICOM_11801/"                                                                                  # set dataset path
    train_img_dir             = DATASET+"input_data_splitted/train/images/"                                                 # set original image path
    train_mask_dir            = DATASET+"input_data_splitted/train/masks/"                                                  # set mask image path
    train_img_list            = os.listdir(train_img_dir)                                                                   # get list of the original image paths
    train_mask_list           = os.listdir(train_mask_dir)                                                                  # get list of the mask image paths
    
    batch_size                = 2                                                                                           # set batch size for testing the script
    
    train_img_datagen         = imageLoader(train_img_dir, train_img_list, train_mask_dir, train_mask_list, batch_size)     # get train image data generator
    
    img, msk                  = train_img_datagen.__next__()                                                                # verify generator.... In python 3 next() is renamed as __next__()
    
    
    img_num                   = random.randint(0,img.shape[0]-1)                                                            # get random integer
    test_img                  = img[img_num]                                                                                # get random original image
    test_mask                 = msk[img_num]                                                                                # get random mask image
 
    plt.figure(figsize=(12, 8))                                                                                             # generate new figure        
    plt.subplot(121)                                                                                                        # generate subplot
    plt.imshow(test_img, cmap='gray')                                                                                       # add orginal image
    plt.subplot(122)                                                                                                        # generate subplot
    plt.imshow(test_mask)                                                                                                   # add mask image
    plt.title('Mask')                                                                                                       # set title of plot
    plt.show()                                                                                                              # show plot

    
    sub_imgs  = divide_image(test_mask)                                     # divide the test image                                                                                                                                           
    for sub_img in sub_imgs:                                                # enumarate the sub images
        plt.figure()                                                        # generate new figure
        plt.imshow(sub_img,cmap='gray')                                     # show the sub image
        
    join_img  = join_image(sub_imgs)                                        # join the sub images
    plt.figure()                                                            # generate new figure
    plt.imshow(join_img,cmap='gray')                                        # show the joined image
    img_ci,sub_imgs_ci = compare_class_imbalance_ratio(test_mask,sub_imgs)  # compare class imbalance ratio for the test image and sub images    
    print(f"image class imbalance ratio: {img_ci}\nclass imbalance ratio for sub images: {sub_imgs_ci} \n mean of the class imbalance ratio for non-zero sub images : ",np.mean([sub_imgs_ci!=0.0]))  # print the compared result
