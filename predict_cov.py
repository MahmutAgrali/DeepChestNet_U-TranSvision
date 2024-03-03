# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 15:34:36 2022
"""

from numpy import load
from matplotlib import pyplot
import os 
import cv2
import math
from tensorflow.keras.models import load_model
import config
import numpy as np

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

# load and prepare training images
def load_real_samples(filename):
	# load compressed arrays
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

# select a batch of random samples, returns images and target
def generate_real_samples(trainA,trainB, ix1):
    X1, X2 = np.expand_dims(trainA[ix1],axis=0), np.expand_dims(trainB[ix1],axis=0)
    return [X1, X2]
 
# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples):
	# generate fake instance
    X = g_model.predict(samples)
    return X



if __name__ == "__main__":
    import pandas as pd # import pandas
    # Define dataset paths and extensions
    DATASET                     = config.DATASET                       # get dataset path from config
    DATASET_ORIGINAL_DICOM_PATH = config.DATASET_ORIGINAL_DICOM_PATH   # get dataset orginal dicom path from config
    DATASET_ORIGINAL_JPG_PATH   = config.DATASET_ORIGINAL_JPG_PATH     # get dataset original jpg path from config
    DATASET_MASK_PATH           = config.DATASET_COV_MASK_PATH             # get dataset mask path from config
    DATASET_ORIGINAL_DICOM_EXT  = config.DATASET_ORIGINAL_DICOM_EXT    # get dataset dicom extension from config
    DATASET_ORIGINAL_JPG_EXT    = config.DATASET_ORIGINAL_JPG_EXT      # get dataset jpg extension from config
    DATASET_MASK_EXT            = config.DATASET_MASK_EXT              # get dataset mask png extension from config
    IMAGE_DIR_NPY               = config.IMAGE_DIR_NPY                 # get original image path of file that has npy extension
    MASKS_DIR_NPY               = config.MASKS_DIR_NPY                 # get mask image path of file that has npy extension
    results_dir                 = config.results_dir                   # get results directory
    SAVE_BALANCED_DATA          = config.SAVE_BALANCED_DATA
    
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
                                "Annotions"])                             # generate dataframe using the parameters
    df.to_excel(DATASET+results_dir+"all_results.xlsx",index=False)       # save the df to the xlsx file



# load image data
dataset = load_real_samples('cov_256.npz')
g_model = load_model("model_256.h5") 

idx_file_name = 0

trainA, trainB = dataset
# batch = 2
for i in range(int(math.ceil(trainA.shape[0]))):
    [X_realA, X_realB] = generate_real_samples(trainA, trainB, i)

    X_fakeB = generate_fake_samples(g_model, X_realA)

    X_realA[0,:,:,0] = (X_realA[0,:,:,0] + 1) / 2.0
    X_realB[0,:,:,0] = (X_realB[0,:,:,0] + 1) / 2.0
    X_fakeB[0,:,:,0] = (X_fakeB[0,:,:,0] + 1) / 2.0

    # cv2.imshow("X_fakeB", X_fakeB[0,:,:,0])                     # show image
    # cv2.imshow("X_realB", X_realB[0,:,:,0])                     # show image
    # cv2.imshow("X_realA", (X_realA[0,:,:,0])*255)               # show image
    # cv2.waitKey(0)  

    np.save(IMAGE_DIR_NPY+'image_orginal_'+str(idx_file_name)+'.npy', X_realB[j,:,:,0])    # Save image to npy
    np.save(MASKS_DIR_NPY+'mask_orginal_'+str(idx_file_name)+'.npy', X_realA[j,:,:,0])     # Save image to npy

    np.save(IMAGE_DIR_NPY+'image_generated_'+str(idx_file_name)+'.npy', X_fakeB[j,:,:,0])  # Save image to npy
    np.save(MASKS_DIR_NPY+'mask_generated_'+str(idx_file_name)+'.npy', X_realA[j,:,:,0])   # Save image to npy
    
    idx_file_name +=1

    print("[{} \%],left [{}]".format((i/trainA.shape[0])*100,trainA.shape[0]-i))

