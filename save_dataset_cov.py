# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 10:48:34 2022
"""

from os import listdir
from numpy import asarray
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from numpy import savez_compressed
from pydicom import dcmread
from skimage.transform import resize
import cv2

def dicom_to_JPG(dcm_img):
	"""Convert dicom image format to JPG image format."""
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

    new_img = new_img.astype(np.uint8)
	return new_img


# load all images in a directory into memory
def load_images(path_img, path_mask):
	src_list, tar_list = list(), list()

	# enumerate filenames in directory, assume all are images
	counter = 0
	for filename in listdir(path_img):
		# load and resize the image
		img = np.array(dicom_to_JPG(dcmread(path_img + filename)))

		mask = img_to_array(load_img(path_mask+filename.replace(".dcm",".png")))

		# split into satellite and map
		img   = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)    			# Resize images to 256x256 
		mask  = resize(mask, (256, 256), mode='constant', preserve_range=True)                  # Resize images to 256x256
		mask = np.moveaxis([mask[:,:,0],mask[:,:,0],mask[:,:,0]],0,-1)
		img = np.moveaxis([img,img,img],0,-1)
		# cv2.imshow("img", img[:,:,0])                       # show image
		# cv2.imshow("mask", mask[:,:,0])                     # show mask image

		src_list.append(mask)
		tar_list.append(img)
		counter +=1
		print(counter)

	return [asarray(src_list), asarray(tar_list)]
 
# dataset path
PATH_IMG = "Covid19Project/CODING/DATASETS/MAIN_DATASETS/COVID_DATASET_2/original/" 
PATH_MASK = "Covid19Project/CODING/DATASETS/MAIN_DATASETS/COVID_DATASET_2/mask/" 

# load dataset
[src_images, tar_images] = load_images(PATH_IMG,PATH_MASK)
print('Loaded: ', src_images.shape, tar_images.shape)

# save as compressed numpy array
filename = 'cov_256.npz'
savez_compressed(filename, src_images, tar_images)
print('Saved dataset: ', filename)