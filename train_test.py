# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 13:12:17 2021

@ Code to train batches of cropped Covid19 images using 2D U-net.

@ Please get the data ready and define custom data gnerator using the other files in this directory.

@ Images are expected to be 256x256x1 npy data (1 corresponds to the 1 channels of the image that is gray scale)

@ Masks are expected to be 256x256x1 npy data (3 corresponds to the 3 classes / labels)

@ You can change input image sizes to customize for your computing resources.

############################################

@@ NOTE : You can change the code 'keras.utils.generic_utils.get_custom_objects().update(custom_objects)' in the directory (*) to 'keras.utils.get_custom_objects().update(custom_objects)', if there any error such as
        'There is no module generic_utils'.
        
      *      ./anaconda3/envs//Lib/site-packages/efficientnet/__init__.py     

"""

#####################################################
# @ import required modules
#####################################################
from cProfile import label
import os
import copy
import numpy as np
from data_generator import *
import keras
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard, EarlyStopping, ReduceLROnPlateau,CSVLogger
import matplotlib
from matplotlib import pyplot as plt
import glob
import random
from tensorflow.random import set_seed
import tensorflow_addons as tfa
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
from skimage import exposure,filters,segmentation,morphology

import tensorflow as tf
from shutil import move
from tensorflow.keras.models import load_model
import pandas as pd
import segmentation_models as sm
from  models import *
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import config
from sklearn.model_selection import KFold
from tensorflow.keras import backend as K
import datetime
import shutil
from sklearn.model_selection import train_test_split


def reset_seeds(reset_graph_with_backend=True):
    """
    @ Function to reset random number generators' seeds for reproducibility.
    @ params :
        @@ reset_graph_with_backend: If True, reset Keras and TensorFlow graphs.
    @ return : 
       @@
    """
    
    # Reset Keras and TensorFlow graphs if requested
    if reset_graph_with_backend is not None:
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        print("KERAS AND TENSORFLOW GRAPHS RESET")  # Optional message

    # Reset random number generators' seeds
    np.random.seed(SEED_NUMBER)
    random.seed(SEED_NUMBER)
    tf.compat.v1.set_random_seed(SEED_NUMBER)
    set_seed(SEED_NUMBER)  
        
    print("RANDOM SEEDS RESET")  # Optional message
    

SEED_NUMBER = config.SEED_NUMBER  # get seed number

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(SEED_NUMBER)


os.environ['PYTHONHASHSEED'] = str(SEED_NUMBER)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
random.seed(SEED_NUMBER)

# The below set_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
set_seed(SEED_NUMBER)           


reset_seeds()

def regex(mydict, reg):
    """
    @  Function to filter dictionary items based on keys that start with a specified prefix.
    @ params :
     	mydict: Dictionary to filter.
    	reg: Prefix or regular expression pattern to match keys.

    @ return:
       @@ If a match is found, returns the corresponding value.
       @@ If no match is found, returns an empty list.
    """
    li = list()

    # Iterate through key-value pairs in the dictionary
    for k, v in mydict.items():
        # Check if the key starts with the specified prefix
        if k.startswith(reg):  # or use a regex pattern here
            return v  # Return the corresponding value for the first match found
            li.append(v)  # This line is unreachable; it will never be executed

    # If no match is found, return an empty list
    return li
 
#####################################################

class Covid19():    
        
    ########################################################
    # This part allow to get the config from config.py
    #######################################################
    DATASET        = config.DATASET                    # get dataset path
    train_img_dir  = config.train_img_dir              # get train image path
    train_mask_dir = config.train_mask_dir             # get train mask path
    results_dir    = config.results_dir                # get results path
    tmp_dir        = config.tmp_dir                    # get temp path
    models_dir     = config.models_dir                 # get model path
    
    blind_test_img_dir  = config.blind_test_img_dir     # get blind test image path 
    blind_test_mask_dir = config.blind_test_mask_dir    # get blind test mask path  
    blind_test_file     = config.blind_test_file        # get blind test file path  
    DATASET_SPLIT_BLIND_TEST_FILE = config.DATASET_SPLIT_BLIND_TEST_FILE
    DATASET_SPLIT = config.DATASET_SPLIT

    val_img_dir  = config.val_img_dir    # get validation image path
    val_mask_dir = config.val_mask_dir   # get validation mask path
    
    img_list = os.listdir(train_img_dir)    # get list of the train image paths
    msk_list = os.listdir(train_mask_dir)   # get list of the train mask paths
    
    num_images          = len(os.listdir(train_img_dir))    # get number of the image in the training dataset
    EPOCHS              = config.EPOCHS                 # get epoch 
    BATCH_SIZE          = config.BATCH_SIZE             # get batch size
    ARCH                = config.ARCH                   # get architecture
    OPTIMIZER           = config.OPTIMIZER              # get optimizer
    MOMENTUM            = config.MOMENTUM               # get momentum
    LEARNING_RATE       = config.LEARNING_RATE          # get learning rate
    KERNEL_INITIALIZER  = config.KERNEL_INITIALIZER     # get kernel initializer
    SHOW_PLT            = config.SHOW_PLT               # get SHOW_PLT variable
    IS_GPU              = config.IS_GPU                 # get IS_GPU variable
    SEED_NUMBER         = config.SEED_NUMBER            # get seed number
    GPU_NUM             = config.GPU_NUM
    
    if(not SHOW_PLT):         # check to show the plot
        matplotlib.use('Agg') # use to save the figure without showing the figure 

    
 
    def __init__(self):
        """
        @ This method is initialize method for the class
        @ params :
            @@
        @ return : 
            @@
        """
        global iou_c
        iou_c = 0

        os.environ["CUDA_VISIBLE_DEVICES"]= str(self.GPU_NUM)            # set GPU usage is false
        os.environ["TF_CUDNN_DETERMINISTIC"]= "1"                        # set GPU usage is false
        os.environ["TF_DETERMINISTIC_OPS"]= "1"                          # set GPU usage is false
      
        self.gpu_list = tf.config.list_physical_devices('GPU')
        self.cpu_list = tf.config.list_physical_devices('CPU')
        print("GPU: {}, CPU: {}".format(self.gpu_list,self.cpu_list))

        
    def check_gpu(self):
        """
        @ This method checks to run on GPU or not 
        @ params: 
            @@ 
        @ return: 
            @@
        """
        if(self.IS_GPU):                                                # check to run the GPU device
            gpus = tf.config.experimental.list_physical_devices('GPU')  # get all GPU devices
            for gpu in gpus:                                            # get GPUs one by one 
                print("Name:", gpu.name, "  Type:", gpu.device_type)    # get GPU name
                tf.config.experimental.set_memory_growth(gpu, True)     # set GPU usage is true
        else:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"              # set GPU usage is false
            os.environ["CUDA_VISIBLE_DEVICES"]="-1"                     # set GPU usage is false

    def specificity(self, y_true, y_pred):
        """
        @ Calculate specificity, a measure of a binary classification model's ability to correctly identify negative instances.

        @ params:
            @@ y_true: True binary labels. Type: np.array
            @@ y_pred: Predicted binary labels. Type: np.array

        @ return:
            @@specificity: Specificity score. Type: float
        """
        def r(y_true, y_pred):
            """
            @ Calculate specificity for binary classification.

            @ params:
                @@ y_true: True binary labels. Type: np.array
                @@ y_pred: Predicted binary labels. Type: np.array

            @ return:
                @@ specificity: Specificity score. Type: float
            """
            # Calculate negative labels
            neg_y_true = 1 - y_true
            neg_y_pred = 1 - y_pred

            # Calculate false positives (fp) and true negatives (tn)
            fp = K.sum(neg_y_true * y_pred)
            tn = K.sum(neg_y_true * neg_y_pred)

            # Calculate specificity
            specificity = tn / (tn + fp + K.epsilon())
            return specificity

        # Return the specificity score calculated by the inner function
        return r(y_true, y_pred)

    def dice_coef(self,y_true, y_pred, smooth=1):
        """
        @ This method provides to get dice coefficient
        @ params : 
            @@ y_true   : the true image labelled image :: np.array
            @@ y_pred   : the predicted labelled image :: np.array
            @@ smooth=1 : the smooth factor :: integer
        @ returns : 
            @@ dice_coef : the dice coefficient :: float
        """
      
        y_true_f     = K.flatten(y_true)                                                   # get vector of the y_true matrix
        y_pred_f     = K.flatten(y_pred)                                                   # get vector of the y_pred matrix
        intersection = K.sum(y_true_f * y_pred_f)                                          # get intersection of the two vectors
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth) # return dice coefficient 
                                      
   
     
    def dice_coef_loss(self,y_true, y_pred):
        """
        @ This method provides to get loss of the dice coefficient 
        @ params : 
            @@ y_true : the true image labelled image :: np.array
            @@ df     : the predicted labelled image :: np.array
        @ returns : 
            @@ dice_coef_loss : the loss of the dice coefficient :: float
        """
        return -self.dice_coef(y_true, y_pred)

       
    def append_to_excel(self,fpath, df):
        """
        @ This method appends the history to the xlsx file
        @ params : 
            @@ fpath : the file path :: string
            @@ df    : the dataframe :: pandas.Dataframe
        @ returns : 
            @@ 
        """
       
        book  = load_workbook(fpath)                                    # load xlsx file
        sheet = book.get_sheet_by_name(book.get_sheet_names()[0])       # get the sheet
        
        for row in dataframe_to_rows(df, index=False, header=False):    # load the dataframe into the sheet
            sheet.append(row)                                           # append the row to the sheet
        
        book.save(fpath)                                                # save the modified excel at the desired location    

        
    def make_path(self):
        """
        @ This method provides to make a directory 
        @ params : 
            @@ 
        @ returns : 
            @@ 
        """
        path = self.path + "/"                   # add to path '/'
        while True:                              # for infinity loop 
            if not os.path.exists(path):         # check path is exist
                 break                           # break from the while
            path = self.path                     # store the path in the local variable
            path = path + "_hash_"+str(hash(random.randint(0, 100000000000)))+str(hash(random.randint(0, 100000000000)))+"/" # add random hash to the path
            
        os.mkdir(path)    # generate path
        self.path = path  # set the local path to the global path
    
    def get_dict_index(self,dictionary,value):
        """
        @ This method get index of dictionary
        @ params : 
             @@ dictionary :: a dictinary list              :: dict
             @@ value      :: value for desired index       :: integer, float or string
        @ return : 
            @@
        """    
        list_ = [i for i, d in enumerate(dictionary) if d==value ] # find index in the dictionary
        return list_[-1]                                           # return last element of the index list   
        
    def save_log_model(self):
        """
        @ This method saves the history to xlsx file, the model to hdf5 files and returns the path
        @ params : 
            @@ 
        @ return : 
            @@ 
        """
       
        max_iou_score        = max(self.history[self.supervision+'iou_score'])                                          # get max iou score
        max_val_iou_score    = max(self.history['val_'+self.supervision+'iou_score'])                                   # get max val iou score
        max_dsc_score        = max(self.history[self.supervision+'dice_coef'])                                          # get max dsc score
        max_val_dsc_score    = max(self.history['val_'+self.supervision+'dice_coef'])                                   # get max val dsc score
        max_miou_score       = max(self.history[self.supervision+'miou_score'])                                         # get max miou score
        max_val_miou_score   = max(self.history['val_'+self.supervision+'miou_score'])                                  # get max miou dsc score
    
        max_iou_score_in     = self.get_dict_index(self.history[self.supervision+'iou_score'],max_iou_score) + 1        # get index of max iou score
        max_val_iou_score_in = self.get_dict_index(self.history['val_'+self.supervision+'iou_score'],max_val_iou_score) + 1# get index of max val iou score
        max_iou_score        = round(max_iou_score*100,2)                                              # get max iou score
        max_val_iou_score    = round(max_val_iou_score*100,2)                                          # get max val iou score
        max_dsc_score        = round(max_dsc_score*100,2)                                              # get max dsc score
        max_val_dsc_score    = round(max_val_dsc_score*100,2)                                          # get max val dsc score
        max_miou_score       = round(max_miou_score*100,2)                                             # get max miou score
        max_val_miou_score   = round(max_val_miou_score*100,2)                                         # get max val miou score
      
        self.sub_path = "logs_{}_epoch_{}_batch_{}_arch_{}_iou_score".format(self.EPOCHS,self.BATCH_SIZE,self.ARCH,max_iou_score) # get sub path
        self.path     = self.DATASET+"results/" + self.sub_path                                                                   # get path
        
        self.make_path()                               # generate the path
       
        file_name = self.path + self.sub_path +".xlsx" # get file name for the xlsx file
        
         
        
        os.mkdir(self.path + self.models_dir)                                                        #  generate path for models that is saved in each epoch
        saved_models = sorted(glob.glob(self.DATASET + self.results_dir +  self.tmp_dir + "*.tf"))   # get saved models or log of the tensorboard that is in the tmp directory
        logs_path = self.DATASET + self.results_dir +  self.tmp_dir + "logs"                         # get saved models or log of the tensorboard that is in the tmp directory
        move(logs_path, self.path + self.models_dir)
        move(self.DATASET + self.results_dir +  self.tmp_dir+"model_history_log.csv", self.path + self.sub_path)

        for _,path_ in enumerate(saved_models):                                                   # enumarate the saved models list
            _,file_n = os.path.split(path_)                                                       # split file name                        
            in_ = int(file_n.split('-')[1])                                                       # get index for saved model
            if(max_iou_score_in != in_ and max_val_iou_score_in != in_):                          # check index    
                shutil.rmtree(path_)
            else:
                move(path_, self.path + self.models_dir)                                             # move the models to the specified result directory
                if max_val_iou_score_in == in_:
                    test_model_path = path_
        test_model_path = test_model_path.replace("/","\\").replace("tmp",self.models_dir)
        test_model_path = os.path.split(test_model_path)
        try:
            self.blind_test_last_epoch(self.path + self.models_dir+test_model_path[1])
        except Exception as e:
            print("An error occured during the blind test ...",e)
        
        df  = pd.DataFrame(self.history)               # generate dataframe using the history of the training
        
        try:
            df1 = pd.DataFrame({"Epoch":[max(max_iou_score_in,max_val_iou_score_in)],
                                "Batch":[self.BATCH_SIZE],
                                "Architecture":[self.ARCH],
                                "Optimizer":[self.OPTIMIZER],
                                "Learning_Rate":[self.LEARNING_RATE],
                                "Momentum":[self.MOMENTUM],
                                "Kernel_Initializer":[self.KERNEL_INITIALIZER],
                                "Seed_Number":[self.SEED_NUMBER],                        
                                "Max_IOU_Score":[max_iou_score],                     
                                "Max_Val_IOU_Score":[max_val_iou_score],
                                "Max_Test_IOU_Score":[self.history_test[self.supervision+"iou_score"]],
                                "Max_DSC":[max_dsc_score],
                                "Max_Val_DSC":[max_val_dsc_score],
                                "Max_Test_DSC_Score":[self.history_test[self.supervision+"dice_coef"]],
                                "Max_MIOU":[max_miou_score],
                                "Max_Val_MIOU":[max_val_miou_score],
                                "Max_Test_MIOU_Score":[self.history_test[self.supervision+"miou_score"]],
                                "Accuracy":[self.history_test[self.supervision+"accuracy"]],
                                "Recall":[regex(self.history_test,self.supervision+'recall')],
                                "Precision":[regex(self.history_test,self.supervision+'precision')],
                                "Specificity":[regex(self.history_test,self.supervision+'specificity')],
                                "Dataset":[self.DATASET.replace("/", "")],
                                "Date":[datetime.datetime.now().strftime("%d.%m.%Y - %H.%M")],
                                "Annotions":[config.ANNOTIONS]}) # generate dataframe using the parameters
        except:
            df1 = pd.DataFrame({"Epoch":[max(max_iou_score_in,max_val_iou_score_in)],
                                "Batch":[self.BATCH_SIZE],
                                "Architecture":[self.ARCH],
                                "Optimizer":[self.OPTIMIZER],
                                "Learning_Rate":[self.LEARNING_RATE],
                                "Momentum":[self.MOMENTUM],
                                "Kernel_Initializer":[self.KERNEL_INITIALIZER],
                                "Seed_Number":[self.SEED_NUMBER],                        
                                "Max_IOU_Score":[max_iou_score],                     
                                "Max_Val_IOU_Score":[max_val_iou_score],
                                "Max_DSC":[max_dsc_score],
                                "Max_Val_DSC":[max_val_dsc_score],
                                "Max_MIOU":[max_miou_score],
                                "Max_Val_MIOU":[max_val_miou_score],
                                 "Dataset":[self.DATASET.replace("/", "")],
                                "Date":[datetime.datetime.now().strftime("%d.%m.%Y - %H.%M")],
                                "Annotions":[config.ANNOTIONS]}) # generate dataframe using the parameters
        
      

        df = df.append(df1)                                                         # add df1 to df
        df.to_excel(file_name)                                                      # save the df to the xlsx file
        self.append_to_excel(self.DATASET+self.results_dir+"all_results.xlsx", df1) # save df1 to the xksx fşke
        
    def get_optimizer(self):
        """
        @ This method returns the optimizers.
        @ params : 
             @@ 
        @ return : 
            @@
        """
        if(self.OPTIMIZER == 'Adam'):                                                           # check optimizers
            self.optimizer = Adam(learning_rate=self.LEARNING_RATE)                             # get optimizer
        elif(self.OPTIMIZER == 'SGD'):                                                          # check optimizers
            self.optimizer = SGD(learning_rate=self.LEARNING_RATE,momentum=self.MOMENTUM)       # get optimizer
        elif(self.OPTIMIZER == 'RMSprop'):                                                      # check optimizers
            self.optimizer = RMSprop(learning_rate=self.LEARNING_RATE,momentum=self.MOMENTUM)   # get optimizer
        elif(self.OPTIMIZER == 'Adadelta'):                                                     # check optimizers
            self.optimizer = Adadelta(learning_rate=self.LEARNING_RATE)                         # get optimizer
        elif(self.OPTIMIZER == 'Adagrad'):                                                      # check optimizers
            self.optimizer = Adagrad(learning_rate=self.LEARNING_RATE)                          # get optimizer
        elif(self.OPTIMIZER == 'Adamax'):                                                       # check optimizers
            self.optimizer = Adamax(learning_rate=self.LEARNING_RATE)                           # get optimizer
        elif(self.OPTIMIZER == 'Nadam'):                                                        # check optimizers
            self.optimizer = Nadam(learning_rate=self.LEARNING_RATE)                            # get optimizer
        elif(self.OPTIMIZER == 'Ftrl'):                                                         # check optimizers
            self.optimizer = Ftrl(learning_rate=self.LEARNING_RATE)                             # get optimizer
        else:
            raise AssertionError("Please select from the list of the optimizer. {} is not valid optimizer".format(self.OPTIMIZER)) # raise the error

    
    def get_dataset(self):
        """
        @ This method loads the training/validation dataset.
        @ params : 
             @@ 
        @ return : 
            @@
        """      
        self.train_img_list      = os.listdir(self.train_img_dir)            # get train original images 
        self.train_mask_list     = os.listdir(self.train_mask_dir)           # get train original masks
    
   
        
    def get_model(self):
        """
        @ This method gets the desired model.  
        @ params : 
             @@ 
        @ return : 
            @@
        """ 
         
        if self.ARCH.__eq__("UNet_plus_plus"):                                                                                                    # check architecture for UNet_plus_plus
         self.model  = UNet_plus_plus(256, 256, 1)                                                                                                  # generate UNet_plus_plus model

        elif self.ARCH.__eq__("128"):                                                                                                       # check architecture for unet
            self.model = unet_2d_model_128_128(IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1,kernel_initializer=self.KERNEL_INITIALIZER)    # generate unet model
        
        elif self.ARCH.__eq__("U_NET"):                                                                                                              # check architecture for U_NET
            self.model = U_NET(IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1,kernel_initializer=self.KERNEL_INITIALIZER)                             # generate U_NET model

        elif self.ARCH.__eq__("DeeplabV3Plus_vgg19"):
           self.model = DeeplabV3Plus_vgg19(IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1)          # generate DeeplabV3Plus_vgg19 model
        
        elif self.ARCH.__eq__("DeeplabV3Plus"):
           self.model = DeeplabV3Plus(IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1)          # generate DeeplabV3Plus model

        elif self.ARCH.__eq__("DeeplabV3Plus_vgg16"):
           self.model = DeeplabV3Plus_vgg16(IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1)          # generate DeeplabV3Plus_vgg19 model
        
        elif self.ARCH.__eq__("DeepLabV3Plus_resnet50"):
            self.model = DeeplabV3Plus_resnet50(IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1)          # generate DeepLabV3Plus_resnet50 model
        
        elif self.ARCH.__eq__("dense_unet_2d"):
            self.model = dense_unet_2d(IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1)          # generate unet model 

        elif self.ARCH.__eq__("ResUnet"):                                      8                                                        # check architecture for ResUnet
            resunet = ResUnet(IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1)                                                            # generate ResUnet model
            self.model = resunet.get_model()  
          

        elif self.ARCH.__eq__("UNet_vgg16"):                                                                                                   # check architecture for vgg16-unet
            self.model = UNet_vgg16(IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1,kernel_initializer=self.KERNEL_INITIALIZER)                  # generate vgg16-unet model      
                           
        elif self.ARCH.__eq__("UNet_vgg19"):                                                                                                   # check architecture for vgg19-unet
            self.model = UNet_vgg19(IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1,kernel_initializer=self.KERNEL_INITIALIZER)                  # generate vgg19-unet model                         
        
        else: 
            raise AssertionError("Please select from the list of the arhitecture. {} is not valid arhitecture".format(self.ARCH))         # raise the error
        if supervision:
            self.supervision = "outputs_"
        else:
            self.supervision = ""

    
    def get_data_gen(self,i_train=None,i_test=None,k_fold=False):
        """
        @ This method gets the data generator.
        @ params : 
             @@ i_train=None :: index for train k'th fold            :: integer
             @@ i_test=None  :: index for validation k'th fold       :: integer
             @@ k_fold=False :: set status for k_fold, open or close :: bool
        @ return : 
            @@
        """    
      
        if(k_fold and type(i_train) != type(None) and type(i_test) != type(None)):       # check k_fold is true
            X_train, X_val, y_train, y_val = train_test_split(np.array(self.train_img_list)[i_train], np.array(self.train_mask_list)[i_train], test_size=0.2, random_state=42)

            self.train_img_datagen = imageLoader(self.train_img_dir, X_train,self.train_mask_dir, y_train, self.BATCH_SIZE,k_fold=True,img_dir2=self.DATASET_SPLIT+"gen/")   # generate data generator for train dataset
            self.val_img_datagen   = imageLoader(self.train_img_dir, X_val, self.train_mask_dir, y_val, self.BATCH_SIZE)    # generate data generator for validation dataset
            self.blind_test_img_datagen = imageLoader(self.train_img_dir, np.array(self.train_img_list)[i_test], self.train_mask_dir, np.array(self.train_mask_list)[i_test], self.BATCH_SIZE)    # generate data generator for validation dataset
            
            self.steps_per_epoch     = len(X_train)//self.BATCH_SIZE # calculate steps per epoch for training dataset 
            self.val_steps_per_epoch = len(X_val)//self.BATCH_SIZE   # calculate steps per epoch for validation dataset

        elif((not k_fold) and (type(i_train) ==  type(None)) and (type(i_test) ==  type(None))):                                                     # check k_fold is false
            self.train_img_datagen = imageLoaderSpervision(self.train_img_dir, self.train_img_list,self.train_mask_dir, self.train_mask_list, self.BATCH_SIZE) # generate data generator for train dataset
            self.val_img_datagen = imageLoaderSpervision(self.val_img_dir, self.val_img_list, self.val_mask_dir, self.val_mask_list, self.BATCH_SIZE)          # generate data generator for validation dataset
            
        else:
            raise AssertionError("Please don't use i_train or i_test, if k_fold is False") # raise the error
                
    def iou_score(self,y_true, y_pred):
        """
        @ This method gets the data generator.
        @ params : 
             @@ y_true :: the labelled image (true image) :: np.array
             @@ y_pred :: the predicted image             :: np.array
        @ return : 
            @@ iou_score :: the iou score :: float32
        """   
        def f(y_true, y_pred):
            global iou_c
           
            intersection = (y_true * y_pred).sum()                     # get intersection y_true and y_pred
            union        = y_true.sum() + y_pred.sum() - intersection  # get union y_true and y_pred
            x            = (intersection + 1e-15) / (union + 1e-15)    # get iou score
            x            = x.astype(np.float32)                        # convert iou score from float64 to float32
            # print(x)
            return x                                                   # return iou score
        return tf.numpy_function(f, [y_true, y_pred], tf.float32)      # return iou score
    
       
    def get_metrics(self):
        """
        @ This method gets the data generator.
        @ params : 
             @@ 
        @ return : 
            @@
        """   
        self.dice_loss  = sm.losses.DiceLoss()                                      # generate Dice Loss 
        self.focal_loss = sm.losses.CategoricalFocalLoss()                          # generate Categorical Focal Loss
        self.total_loss = self.dice_loss + (1 * self.focal_loss)                    # get total of the Dice and Categorical Focal Loss, 'binarycrossetropy'
        
        self.metric_miou = keras.metrics.MeanIoU(num_classes=2,name="miou_score")       # generate mean iou score metric
        self.metrics = {"binary_crossentropy":"binary_crossentropy","dice_loss_plus_1focal_loss":self.total_loss,"accuracy":"accuracy", "recall":tf.keras.metrics.Recall(), "precision":tf.keras.metrics.Precision(),"specificity":self.specificity,"iou_score":self.iou_score,"miou_score":self.metric_miou,"dice_coef":self.dice_coef} 

    def model_compile_and_fit(self):
        """
        @ This method provides to compile and fit the model.
        @ params : 
             @@ 
        @ return : 
            @@
        """   
                 
        if (not hasattr(self,"optimizer") or not hasattr(self,"metrics") or not hasattr(self,"model") or not hasattr(self,"train_img_datagen")): # check the variables if exists
            raise AssertionError("Please call first get_optimizer() or get_metrics() or get_model() or get_data_gen()")                          # raise the error
        self.model.compile(optimizer = self.optimizer, loss=self.total_loss, metrics=list(self.metrics.values()))                                # compile the model 
       

      
        print(self.model.summary())                                                                               # print the summary
        print(self.model.input_shape)                                                                             # print the input shape
        print(self.model.output_shape)                                                                            # print the output shape
        
        if self.supervision == "outputs_":
            check_point_dir = self.DATASET + self.results_dir +  self.tmp_dir + "saved_model-{epoch:02d}-{epoch:02d}-{outputs_iou_score:.2f}-{val_outputs_iou_score:.2f}.tf"#hdf5"    # get check point path
        else:
            check_point_dir = self.DATASET + self.results_dir +  self.tmp_dir + "saved_model-{epoch:02d}-{epoch:02d}-{iou_score:.2f}-{val_iou_score:.2f}.tf"#hdf5"    # get check point path
 
        checkpoint      = ModelCheckpoint(check_point_dir, monitor='val_'+self.supervision+'iou_score', verbose=1, save_best_only=True, mode='max')   # generate model check point for save the model at the each epoch
        csv_logger = CSVLogger(self.DATASET + self.results_dir +  self.tmp_dir+"model_history_log.csv", append=True)
        
        callbacks       = [checkpoint,
                          csv_logger,
                        EarlyStopping(patience=10, monitor='val_'+self.supervision+'loss',restore_best_weights=True),
                      
                        TensorBoard(log_dir=self.DATASET + self.results_dir +  self.tmp_dir + 'logs'),
                      
                        ] # get callbacks

        self.history    = self.model.fit(self.train_img_datagen,
                          steps_per_epoch=self.steps_per_epoch,
                          epochs=self.EPOCHS,
                          verbose=1,
                        #   use_multiprocessing=True,
                          validation_data=self.val_img_datagen,
                          validation_steps=self.val_steps_per_epoch,
                          callbacks=[callbacks],
                          shuffle=False)                # fit the model  
        self.history     = self.history.history         # get the history dictionary
    
    def save_plot(self,fig_name_loss,fig_name_iou,is_train=True):
        """
        @ This method provides to save the figure.
        @ params : 
             @@ fig_name_loss  :: the figure name for the loss           :: String
             @@ fig_name_iou   :: the figure name for iou score          :: String
             @@ is_train=True  :: set status for is_train, true or false :: bool
        @ return : 
            @@
        """   
        if (not hasattr(self,"history")):                                                                # check the history variable if exists
            raise AssertionError("Please call first model_compile_and_fit() or blind_test_each_epoch()") # raise the error

        self.loss      = self.history[self.supervision+'loss']       # get training loss
        
        self.iou_score_ = self.history[self.supervision+'iou_score']  # get training iou score
        
        fig_ = [[self.loss,'','Testing loss','','Testing Loss',fig_name_loss],[self.iou_score_,'','Testing IOU Score','','Testing IOU Score',fig_name_iou]]  # get labels and scores

        if(is_train):                                          # check the training status                   
            self.val_loss      = self.history['val_'+self.supervision+'loss']      # get validation loss
            self.val_iou_score = self.history['val_'+self.supervision+'iou_score'] # get validation iou score
        
            fig_ =  [[self.loss,self.val_loss,'Training loss','Validation loss','Training and Validation Loss',fig_name_loss],[self.iou_score_,self.val_iou_score,'Training IOU Score','Validation IOU Score','Training and Validation IOU Score',fig_name_iou]]   # get labels and scores
    
                                                                                                             
        for index, [line1,line2, label1,label2, title,fig_name] in enumerate(fig_) :   # enumarate fig_ list
            epochs = range(1, len(line1) + 1)                                          # generate epoch range

            plt.figure()                                    # generate new figure for plotting
            plt.plot(epochs, line1, 'y', label=label1)      # plot training loss or iou score
            if(is_train):                                   # check the training status
                plt.plot(epochs, line2, 'r', label=label2)  # plot validation loss  or iou score
            plt.title(title)                                # set the title to the plot
            plt.xlabel('Epochs')                            # set the label to the plot
            if(index == 0):                                 # check the index
                plt.ylabel('Loss')                          # set the label to the plot
            else:
                plt.ylabel('IOU Score')                     # set the label to the plot
            plt.legend()                                    # set the legend to the plot
            plt.savefig(fig_name)                           # save the figure 
            if self.SHOW_PLT:                               # check to show figure
                plt.draw()                                  # show figure

    def show_predicted_image(self,blind_test_img,blind_test_mask,pred_blind_test):
      """
         @ This method provides to show the predicted image.
         @ params : 
              @@ blind_test_img    :: the bilind test image                :: np.array
              @@ blind_test_mask   :: the blind test mask                  :: np.array
              @@ pred_blind_test   :: the predicted image (in blind test)  :: np.array
         @ return : 
             @@
      """   
      self.fig.canvas.flush_events()            # clear the figure
      sp1 = self.fig.add_subplot(231)           # generate subplot
      sp1.set_title('Testing Image')            # set the title to the plot
      sp1.imshow(blind_test_img, cmap='gray')   # show the testing original images
      sp2 = self.fig.add_subplot(232)           # generate subplot
      sp2.set_title('Testing Label')            # set the title to the plot
      sp2.imshow(blind_test_mask)               # show the testing mask
      sp3 = self.fig.add_subplot(233)           # generate subplot
      sp3.set_title('Prediction on test image') # set the title to the plot
      sp3.imshow(pred_blind_test)               # show the testing predicted image
      plt.draw()                                # show the figure
      plt.pause(0.01)                           # pause 10 ms
        
    def train(self,i_train=None,i_test=None,k_fold=False):
        """
           @ This method loads the train/validation datasets and trains the model.
           @ params : 
                @@ i_train  :: index of the train k_fold              :: integer
                @@ i_test   :: index of the test k_fold               :: integer
                @@ k_fold   ::  set status for k_fold, true or false  :: bool
           @ return : 
               @@
        """   
     
        if (not k_fold):           # check k_fold status
            self.get_dataset()     # get dataset
            self.get_metrics()     # get metrics
            self.get_optimizer()   # get optimizer
            self.get_model()       # get model
            
        self.get_data_gen(i_train,i_test,k_fold)                                                  # get data generator
        self.model_compile_and_fit()                                                              # compile and fit the model
        self.save_log_model()                                                                     # save history and model                        
        self.save_plot(self.path+self.sub_path+"_loss.png",self.path+self.sub_path+"_iou.png")    # save the plot
        
    
    
    def train_k_fold(self,n_folds = 5):
         """
           @ This method loads the train/validation datasets and trains the model
           @ params : 
                @@               
           @ return : 
               @@
         """   
                  
         k_fold = True                                  # set status for k_fold
         kfold = KFold(n_splits=n_folds, shuffle=True)  # generate KFold 
         self.get_dataset()                             # get dataset
         self.get_metrics()                             # get metrics
         self.get_optimizer()                           # get optimizer
         self.get_model()                               # ge model
  
         
         iou_per_fold, val_iou_per_fold, loss_per_fold, val_loss_per_fold = list(),list(),list(),list()  # generate list 
         fold_no = 1                                                                                     # initialize the fold_no variable
         arch =self.ARCH
         for i_train, i_test in kfold.split(self.train_img_list, self.train_mask_list):  # enumarate each fold
           
             self.ARCH = f"k_{n_folds}_({fold_no})_" +arch
             self.train(i_train,i_test,k_fold)                 # train the model 
             self.ARCH = arch
             
             iou_per_fold.append(self.iou_score_ * 100)        # append the iou score on kth fold
             val_iou_per_fold.append(self.val_iou_score * 100) # append the val iou score on kth fold
             loss_per_fold.append(self.loss)                   # append the loss on kth fold
             val_loss_per_fold.append(self.val_loss)           # append the val loss on kth fold

             fold_no = fold_no + 1  # increase fold number

             
          
    def train_k_fold2(self,n_folds = 5):
        """
        @ This method loads the train/validation datasets and trains the model
        @ params : 
            @@               
        @ return : 
            @@
        """   
                
                                            # get number of fold
        k_fold = True                                  # set status for k_fold
        kfold = KFold(n_splits=n_folds, shuffle=True)  # generate KFold 
        self.get_dataset()                             # get dataset
        self.get_metrics()                             # get metrics
        self.get_optimizer()                           # get optimizer
        self.get_model()                               # ge model

        
        fold_no = 1                                                                                     # initialize the fold_no variable
        arch =self.ARCH
        # with tf.device('/cpu'):

        for i_train, i_test in kfold.split(self.train_img_list, self.train_mask_list):  # enumarate each fold
                # with tf.device('/gpu:0'):
            
            self.ARCH = f"k_{n_folds}_({fold_no})_" +arch
            self.train(i_train,i_test,k_fold)                 # train the model 
            self.ARCH = arch

            fold_no = fold_no + 1  # increase fold number

            
            # print("iou : {} \n val_iou : {} \n loss : {} \n val_loss : {}".format(iou_per_fold,val_iou_per_fold,loss_per_fold,val_loss_per_fold)) # print scores and losses
    
    def initialize_blind_test(self):
        """
          @ This method initializes the blind test and generates the blind test batches
          @ params : 
               @@               
          @ return : 
              @@
        """  
        self.blind_test_img_batch, self.blind_test_mask_batch = self.blind_test_img_datagen.__next__()    # get all testing data using the  data generator.... In python 3 next() is renamed as __next__()
    
    def initialize_blind_test_(self):
        """
          @ This method initializes the blind test and generates the blind test batches
          @ params : 
               @@               
          @ return : 
              @@
        """  
        blind_test_img_list    = os.listdir(self.blind_test_img_dir)   # get the list of the original testing images 
        blind_test_mask_list   = os.listdir(self.blind_test_mask_dir)  # get the list of the mask testing images 
     
        blind_test_img_datagen = imageLoaderSpervision(self.blind_test_img_dir, blind_test_img_list, self.blind_test_mask_dir, blind_test_mask_list, len(blind_test_img_list)) # get testing dataset using datagenerator
       
        self.blind_test_img_batch, self.blind_test_mask_batch = blind_test_img_datagen.__next__()    # get all testing data using the  data generator.... In python 3 next() is renamed as __next__()
        
    def blind_test_each_epoch(self):
        """
          @ This method applies the blind test to the trained saved Unet model  in each epoch
          @ params : 
               @@               
          @ return : 
              @@
        """  
     
        self.initialize_blind_test()     # initialize blind test
        self.get_metrics()               # get metrics
       
        saved_models = sorted(glob.glob(self.DATASET + self.results_dir + self.blind_test_file + "/" +  self.models_dir + "*.tf")) # get saved models
        all_history = list()                                                                                                         # generate the list to save the all history
        
        for _,file in enumerate(saved_models): # enumarete saved models
            print("load model",_)              # print index of the models
            model   = load_model(file,compile=True,custom_objects=self.metrics)                                        # load the model that is at each eposh. For predictions you do not need to compile the model, so ...
            history = model.evaluate(self.blind_test_img_batch,self.blind_test_mask_batch, verbose=1,return_dict=True) # evaluate the metrics for each epoch
            
            loss      = history[self.supervision+'loss']      # get testing loss
            iou_score = history[self.supervision+'iou_score'] # get testing iou score
            
            all_history.append([iou_score,loss]) # add the history to the all_history list
            
        all_history  = np.array(all_history)                                        # convert list to np.array
        self.history =  {"loss":all_history[:][0],"iou_score":all_history[:][1]}    # get history
        self.save_plot(self.DATASET + self.results_dir+self.blind_test_file+"/"+ self.blind_test_file +"_testing_loss.png",self.DATASET + self.results_dir+self.blind_test_file+"/" + self.blind_test_file + "_testing_iou.png",is_train=False) # save plots

    
    def blind_test_last_epoch(self,model_path):
        """
          @ This method applies the blind test to the last trained model
          @ params : 
               @@               
          @ return : 
              @@
        """  
               
        self.initialize_blind_test()  # initialize blind test
        self.get_metrics()            # get metrics
        
    
        try: 
            model   = load_model(model_path,compile=True,custom_objects={"dice_loss_plus_1focal_loss":self.total_loss,"iou_score":self.iou_score,"miou_score":self.metric_miou,"dice_coef":self.dice_coef,"specificity":self.specificity,"TransformerBlock":TransformerBlock,"<lambda>":self.total_loss})  # load the model that is at each eposh. For predictions you do not need to compile the model, so ...
        except:
            try:
                model   = load_model(model_path,compile=True,custom_objects={"dice_loss_plus_1focal_loss":self.total_loss,"iou_score":self.iou_score,"miou_score":self.metric_miou,"dice_coef":self.dice_coef,"TransformerBlock":TransformerBlock,"specificity":self.specificity,"dummy_loss":self.total_loss})  # load the model that is at each eposh. For predictions you do not need to compile the model, so ...
            except:
                model   = load_model(model_path,compile=True,custom_objects={"dice_loss_plus_1focal_loss":self.total_loss,"iou_score":self.iou_score,"miou_score":self.metric_miou,"dice_coef":self.dice_coef,"specificity":self.specificity})  # load the model that is at each eposh. For predictions you do not need to compile the model, so ...

        self.history_test = model.evaluate(self.blind_test_img_batch,self.blind_test_mask_batch, verbose=1,return_dict=True)                                          # evaluate the metrics for last epoch
        print(self.history_test) 
        
       
def run_all(ar_list,k_fold=None):
    """
        @ This method run all ARCHITECTURE
        @ params : 
            @@  ar_list :: architecture list 
            @@ k_fold   :: set True for k-fold approach   
        @ return : 
            @@
    """  

    for ar in ar_list:
        if k_fold==None:
            covid_training = Covid19()  
            print("ARCHITECTURE: ",ar)
            covid_training.ARCH = ar
            covid_training.train()                 # train the model
            del covid_training
        else:
            for j in k_fold:
                covid_training = Covid19()  
                print("ARCHITECTURE: ",ar)
                if ar == "DeepLabV3Plus_resnet50":
                    covid_training.BATCH_SIZE = 16
                elif ar == "DeeplabV3Plus_vgg16" or ar == "DeeplabV3Plus_vgg19":
                        covid_training.BATCH_SIZE = 16
                elif ar == "ResUnet_ori":
                        covid_training.BATCH_SIZE = 16
                elif ar == "ResUnet_plus_plus":
                        covid_training.BATCH_SIZE = 16

                else:
                    covid_training.BATCH_SIZE = 16
                covid_training.ARCH = ar
                covid_training.train_k_fold2(j)
                del covid_training
    
if __name__ == "__main__":
    
    # covid_training = Covid19()             # generate Covid19 class 
    # covid_training.train()                 # train the model
    # run_all(["DeeplabV3Plus_vgg19","DeeplabV3Plus_vgg16","DeepLabV3Plus_resnet50","UNet_plus_plus","UNet_vgg19","UNet_vgg16","U_NET"])
   
    run_all(["DeeplabV3Plus_vgg19","DeeplabV3Plus_vgg16"],[5]) 

    # covid_training.train_k_fold()          # apply k fold method
    # covid_training.blind_test_each_epoch() # apply the blind test on each epoch
    # covid_training.blind_test_last_epoch("./DATASETS/COVID_SEGMENTATION/DATASET_DICOM/results/logs_100_epoch_32_batch_128_arch_80.49_iou_score/models/saved_model-26-26-0.78-0.71.tf") # apply the blind test on the last epoch
    
