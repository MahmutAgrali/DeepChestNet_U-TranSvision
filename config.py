# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 11:31:59 2021

@ This script contains config 

"""
# config that is used by more than one scripts
##################################################################
DATASET_PATH  = "DATASETS/"                                            
DATASET       = DATASET_PATH +"COVID_SEGMENTATION/DATASET_DICOM/"
ANNOTIONS     = ""  #Please write the annotions for each training

INPUT_DIR_NPY =  DATASET+'input_data_npy/'
IMAGE_DIR_NPY = INPUT_DIR_NPY + 'images/'
MASKS_DIR_NPY = INPUT_DIR_NPY + 'masks/'

DATASET_SPLIT            = DATASET + "input_data_splitted/"
DATASET_SPLIT_TRAIN      = DATASET_SPLIT + "train/"
DATASET_SPLIT_VAL        = DATASET_SPLIT + "val/"
DATASET_SPLIT_BLIND_TEST = DATASET_SPLIT + "test/" 
DATASET_SPLIT_BLIND_TEST_FILE = "saved_model"

# config that is used by get_data_ready.py
##################################################################
# Define dataset paths and extensions
DATASET_ORIGINAL_DICOM_PATH = DATASET_PATH + 'MAIN_DATASETS/' + 'COVID_DATASET_2/original/'   # get original dicom path
DATASET_ORIGINAL_JPG_PATH   = DATASET_PATH + 'MAIN_DATASETS/' + 'DATASET_JPG_all/1/'      # get original jpg path
DATASET_COV_MASK_PATH       = DATASET_PATH + 'MAIN_DATASETS/' + 'COVID_DATASET_2/mask/'         # get mask path
DATASET_LUNG_MASK_PATH      = DATASET_PATH + 'MAIN_DATASETS/' + 'LUNG_SEGMENTATION_DATASET/'         # get mask path

DATASET_ORIGINAL_JPG_EXT    = '.jpg'                            # use for jpg
DATASET_ORIGINAL_DICOM_EXT  = '.dcm'                            # use for dicom 
DATASET_MASK_EXT            = '.png'                            # use for png

SAVE_BALANCED_DATA = False

# config that is used by train_test.py
##################################################################
# Define train dataset paths
train_img_dir  = DATASET_SPLIT_TRAIN+"images/"  # get train images path
train_mask_dir = DATASET_SPLIT_TRAIN+"masks/"   # get train masks path

val_img_dir    = DATASET_SPLIT_VAL+"images/"    # get validation images path
val_mask_dir   = DATASET_SPLIT_VAL+"masks/"     # get validation masks path

blind_test_img_dir  =  DATASET_SPLIT_BLIND_TEST+"images/"                 # get blind test images path
blind_test_mask_dir = DATASET_SPLIT_BLIND_TEST+"masks/"                   # get blind test masks path
blind_test_file     = "logs_100_epoch_32_batch_128_arch_80.05_iou_score"  # get blind test file

results_dir = "results/"  # get results path
tmp_dir     = "tmp/"      # get temp path 
models_dir  = "models/"   # get models path 
excel_ext   = ".xlsx"     # get exel extension

SHOW_PLT    = True        # get status of SHOW_PLOT
IS_GPU      = True       # get status of IS_GPU

# Define parameters
EPOCHS             = 100 # 279-100 =179        # get epoch
BATCH_SIZE         = 64 #      # get batch size (The best value is 64)
ARCH               = "128" # ResUnet_fusion(add11)_high_to_low O_net_proposed        # The Unet Architecture that is for 128 x128 images is more better than the other, also 256x256,512x512 and ResUnet architectures are available 
OPTIMIZER          = "Adam"       # Also SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl optimizers are available.
LEARNING_RATE      = 0.0005       # Only default learning rate for SGD is 0.01, others are 0.001, the best is 0.0005
MOMENTUM           = 0.9          # Default momentum is 0.0, only SGD and RMSProp are used
KERNEL_INITIALIZER = 'he_normal'  # The avaliable algorithms are : constant, glorot_normal, glorot_uniform, he_normal, he_uniform, identity, lecun_normal, 
#lecun_uniform, ones, orthogonal, random_normal, random_uniform, truncated_normal, variance_scaling, zeros

SEED_NUMBER        = 1            # get seed number   
GPU_NUM            = 0            # 0 ---> For first GPU 1 --> For second GPU


if OPTIMIZER != "SGD" and OPTIMIZER != "RMSprop":  # check the optimizers are not SGD and RMSProp
    MOMENTUM  = -1                                 # set momentum -1                  

