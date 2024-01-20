# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 13:33:23 2021

Standard 2D Unet and 3D models. 

"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import *#Resizing,Layer,LeakyReLU,AveragePooling2D,SeparableConv2D , TimeDistributed,ConvLSTM2D,Input,Concatenate,Add, Conv2D, MaxPooling2D, multiply, concatenate, BatchNormalization, Dropout, Lambda,UpSampling2D,Conv3D,MaxPooling3D,Conv3DTranspose,Conv2DTranspose,Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from resunet_plus_plus import *
import numpy as np
import cv2 
from skimage import morphology 


alpha = 0.1


#####################################################################


def UNet_plus_plus(img_rows, img_cols, color_type=1, num_class=1, deep_supervision=False):
    dropout_rate = 0.2
    def standard_unit(input_tensor, stage, nb_filter, kernel_size=3):
    
        act = 'elu'
    
        x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv'+stage+'_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
        x = Dropout(dropout_rate, name='dp'+stage+'_1')(x)
        x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv'+stage+'_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
        x = Dropout(dropout_rate, name='dp'+stage+'_2')(x)

        return x
    nb_filter =[16,32,64,128,256] # [32,64,128,256,512]
    act = 'elu'

    bn_axis = 3
    img_input = Input(shape=(img_rows, img_cols, color_type), name='main_input')

    conv1_1 = standard_unit(img_input, stage='11', nb_filter=nb_filter[0])
    pool1 = MaxPool2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = MaxPool2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    up1_2 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up12', padding='same')(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)
    conv1_2 = standard_unit(conv1_2, stage='12', nb_filter=nb_filter[0])

    conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = MaxPool2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    up2_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up22', padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
    conv2_2 = standard_unit(conv2_2, stage='22', nb_filter=nb_filter[1])

    up1_3 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up13', padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=bn_axis)
    conv1_3 = standard_unit(conv1_3, stage='13', nb_filter=nb_filter[0])

    conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
    pool4 = MaxPool2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    up3_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up32', padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
    conv3_2 = standard_unit(conv3_2, stage='32', nb_filter=nb_filter[2])

    up2_3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up23', padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=bn_axis)
    conv2_3 = standard_unit(conv2_3, stage='23', nb_filter=nb_filter[1])

    up1_4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up14', padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=bn_axis)
    conv1_4 = standard_unit(conv1_4, stage='14', nb_filter=nb_filter[0])

    conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])

    up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=bn_axis)
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=bn_axis)
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=bn_axis)
    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])

    nestnet_output_1 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_2)
    nestnet_output_2 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_3)
    nestnet_output_3 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_3', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_4)
    nestnet_output_4 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_4', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_5)

    if deep_supervision:
        model = Model(img_input, [nestnet_output_1,nestnet_output_2,nestnet_output_3,nestnet_output_4])
    else:
        model = Model(img_input, [nestnet_output_4])
    
    return model


#####################################################################

def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=tf.keras.initializers.HeNormal(),
    )(block_input)
    x = BatchNormalization()(x)
    return tf.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output

def DeeplabV3Plus_resnet50(IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS):
    """
    # Estimation_of_Road_Boundary_for_Intelligent_Vehicles_Based_on_DeepLabV3_Architecture.pdf or
    # https://www.researchgate.net/publication/354108791_Estimation_of_Road_Boundary_for_Intelligent_Vehicles_Based_on_DeepLabV3_Architecture
    """
    model_input = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    resnet50 = tf.keras.applications.ResNet50(
        weights=None, include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = UpSampling2D(
        size=(IMG_HEIGHT // 4 // x.shape[1], IMG_WIDTH // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = UpSampling2D(
        size=(IMG_HEIGHT // x.shape[1], IMG_WIDTH // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = Conv2D(1, kernel_size=(1, 1),activation="sigmoid", padding="same")(x)
    return Model(inputs=model_input, outputs=model_output)

def DeeplabV3Plus_vgg16(IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS):
    """
     # Estimation_of_Road_Boundary_for_Intelligent_Vehicles_Based_on_DeepLabV3_Architecture.pdf or
     # https://www.researchgate.net/publication/354108791_Estimation_of_Road_Boundary_for_Intelligent_Vehicles_Based_on_DeepLabV3_Architecture
    """
    model_input = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    vgg16 = tf.keras.applications.VGG16(
        weights=None, include_top=False, input_tensor=model_input
    )
    x = vgg16.get_layer("block5_conv3").output
    x = DilatedSpatialPyramidPooling(x)

    input_e = UpSampling2D(
        size=(IMG_HEIGHT // 4 // x.shape[1], IMG_WIDTH // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_d = vgg16.get_layer("block3_conv3").output
    input_d = convolution_block(input_d, num_filters=256, kernel_size=1)

    x = Concatenate(axis=-1)([input_e, input_d])
    x = convolution_block(x)
    x = convolution_block(x)
    x = UpSampling2D(
        size=(IMG_HEIGHT // x.shape[1], IMG_WIDTH // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = Conv2D(1, kernel_size=(1, 1),activation="sigmoid", padding="same")(x)
    return Model(inputs=model_input, outputs=model_output)

def DeeplabV3Plus_vgg19(IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS):
    """
     # Estimation_of_Road_Boundary_for_Intelligent_Vehicles_Based_on_DeepLabV3_Architecture.pdf or
     # https://www.researchgate.net/publication/354108791_Estimation_of_Road_Boundary_for_Intelligent_Vehicles_Based_on_DeepLabV3_Architecture
    """
    model_input = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    vgg19 = tf.keras.applications.VGG19(
        weights=None, include_top=False, input_tensor=model_input
    )
    x = vgg19.get_layer("block5_conv4").output
    x = DilatedSpatialPyramidPooling(x)

    input_e = UpSampling2D(
        size=(IMG_HEIGHT // 4 // x.shape[1], IMG_WIDTH // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_d = vgg19.get_layer("block3_conv4").output
    input_d = convolution_block(input_d, num_filters=256, kernel_size=1)

    x = Concatenate(axis=-1)([input_e, input_d])
    x = convolution_block(x)
    x = convolution_block(x)
    x = UpSampling2D(
        size=(IMG_HEIGHT // x.shape[1], IMG_WIDTH // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = Conv2D(1, kernel_size=(1, 1),activation="sigmoid", padding="same")(x)
    return Model(inputs=model_input, outputs=model_output)


def dense_unet_2d(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS,kernel_initializer =  'he_uniform'):
    #Build the model for 128x128 and 2d images
    
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    # s = Lambda(lambda x: x / 255)(inputs)
    
    c1 = dense_block(inputs, 128, kernel_initializer)
    
    c2 = transition_block(c1, 64, kernel_initializer)
    c2 = dense_block(c2, 64, kernel_initializer)
    
    c3 = transition_block(c2, 128, kernel_initializer)
    c3 = dense_block(c3, 128, kernel_initializer)
    
    c4 = transition_block(c3, 256, kernel_initializer)
    c4 = dense_block(c4, 256, kernel_initializer)
    
    c5 = transition_block(c4, 512, kernel_initializer)
    c5 = dense_block(c5, 512, kernel_initializer)
    
    
    u_1 = up_sampling(c5, c4, 512)
    
    u_2 = dense_block(u_1, 512, kernel_initializer)
    u_2 = up_sampling(u_2, c3, 256)
    
    u_3 = dense_block(u_2, 256, kernel_initializer)
    u_3 = up_sampling(u_3, c2, 128)
    
    u_4 = dense_block(u_3, 128, kernel_initializer)
    u_4 = up_sampling(u_4, c1, 64)
    
    
    outputs = dense_block(u_4, 64, kernel_initializer)
    outputs = dense_block(outputs, 32, kernel_initializer)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(outputs) 

    model = Model(inputs=[inputs], outputs=[outputs])

    return model 


def up_sampling(input_, c, filter_):
    x = Conv2DTranspose(filter_, (2, 2), strides=(2, 2), padding='same')(input_)
    x = concatenate([x, c], axis=-1)
    return x

def transition_block(input_,filter_,kernel_initializer):
    x = BatchNormalization()(input_)
    x = Conv2D(filter_, (1, 1), activation='relu', kernel_initializer=kernel_initializer, padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    return x

def dense_block(input_,filter_,kernel_initializer):
    
    den_conv_1 = dense_conv_layer(input_, filter_, kernel_initializer)
    den_conv_1_in =Concatenate()([den_conv_1, input_])

    den_conv_2 = dense_conv_layer(den_conv_1_in, filter_, kernel_initializer)
    den_conv_2_in = Concatenate()([den_conv_2, den_conv_1_in])
    
    den_conv_3 = dense_conv_layer(den_conv_2_in, filter_, kernel_initializer)
    den_conv_3_in = Concatenate()([den_conv_3, den_conv_2_in])
        
    den_conv_4 = dense_conv_layer(den_conv_3_in, filter_, kernel_initializer)    
    return den_conv_4

def dense_conv_layer(input_,filter_,kernel_initializer):
    
    x = BatchNormalization()(input_)
    x = Conv2D(filter_, (1, 1), activation='relu', kernel_initializer=kernel_initializer, padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filter_, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(x)
    x = Dropout(0.1)(x)
    return x



def unet_2d_model_128_128(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS,kernel_initializer =  'he_uniform'):
    #Build the model for 128x128 and 2d images
    from keras.backend import tf as ktf

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    # s = Lambda(lambda x: x / 255)(inputs)
 
    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu',  kernel_initializer=kernel_initializer, padding='same')(inputs)
    # c1 = Dropout(0.1)(c1)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu',  kernel_initializer=kernel_initializer, padding='same')(p1)
    # c2 = Dropout(0.1)(c2)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p2)
    # c3 = Dropout(0.2)(c3)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(64, (3, 3), activation='relu',  kernel_initializer=kernel_initializer, padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p3)
    # c4 = Dropout(0.2)(c4)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p4)
    # c5 = Dropout(0.3)(c5)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u6)
    # c6 = Dropout(0.2)(c6)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c6)
     
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u7)
    # c7 = Dropout(0.2)(c7)
    c7 = BatchNormalization()(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c7)
     
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u8)
    # c8 = Dropout(0.1)(c8)
    c8 = BatchNormalization()(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c8)
     
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u9)
    # c9 = Dropout(0.1)(c9)
    c9 = BatchNormalization()(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c9)
     
    outputs = Conv2D(IMG_CHANNELS, (1, 1), activation='sigmoid')(c9) # 'sigmoid'
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model


def UNet_vgg19(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS,kernel_initializer= 'he_uniform'):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    c1 = Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer=kernel_initializer)(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer=kernel_initializer)(c1)
    p1 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(c1)
    
    c2 = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(c2)
    p2 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(c2)
    
    c3 = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(c3)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(c3)
    p3 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(c3)
    
    c4 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(p3)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(c4)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(c4)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(c4)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(c4)
    p4 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(c4)
    
    c5 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(p4)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(c5)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(c5)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(c5)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(c5)
    p5 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(c5)
    
    c6 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(p5)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(c6)
    
    u7 = Conv2DTranspose(512, (2,2), strides=(2,2), padding='same')(c6)
    u7 = concatenate([u7, c5])
    c7 = Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(u7)
    c7 = BatchNormalization()(c7)
    c7 = Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(c7)

    u8 = Conv2DTranspose(512, (2,2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c4])
    c8 = Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(u8)
    c8 = BatchNormalization()(c8)
    c8 = Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(c8)
  
    u9 = Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(c8)
    u9 = concatenate([u9, c3])
    c9 = Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(u9)
    c9 = BatchNormalization()(c9)
    c9 = Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(c9)
 
    u10 = Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c9)
    u10 = concatenate([u10, c2])
    c10 = Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(u10)
    c10 = BatchNormalization()(c10)
    c10 = Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(c10)
                                   
    u11 = Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c10)
    u11 = concatenate([u11, c1])
    c11 = Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(u11)
    c11 = BatchNormalization()(c11)
    c11 = Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(c11)
                                   
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c11)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model


def UNet_vgg16(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS,kernel_initializer= 'he_uniform'):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    c1 = Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer=kernel_initializer)(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer=kernel_initializer)(c1)
    p1 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(c1)
    
    c2 = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(c2)
    p2 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(c2)
    
    c3 = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(c3)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(c3)
    p3 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(c3)
    
    c4 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(p3)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(c4)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(c4)
    p4 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(c4)
    
    c5 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(p4)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(c5)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(c5)
    p5 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(c5)
    
    
    c6 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(p5)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(c6)
    
    u7 = Conv2DTranspose(512, (2,2), strides=(2,2), padding='same')(c6)
    u7 = concatenate([u7, c5])
    c7 = Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(u7)
    c7 = BatchNormalization()(c7)
    c7 = Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(c7)

    u8 = Conv2DTranspose(512, (2,2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c4])
    c8 = Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(u8)
    c8 = BatchNormalization()(c8)
    c8 = Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(c8)
  
    u9 = Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(c8)
    u9 = concatenate([u9, c3])
    c9 = Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(u9)
    c9 = BatchNormalization()(c9)
    c9 = Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(c9)
 
    u10 = Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c9)
    u10 = concatenate([u10, c2])
    c10 = Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(u10)
    c10 = BatchNormalization()(c10)
    c10 = Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(c10)
                                   
    u11 = Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c10)
    u11 = concatenate([u11, c1])
    c11 = Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(u11)
    c11 = BatchNormalization()(c11)
    c11 = Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(c11)
                                   
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c11)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model

class ResUnet():
    def __init__(self,IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_CHANNELS = IMG_CHANNELS
        
    def bn_act(self,x,act=True):
        x = BatchNormalization()(x)
        # x = Dropout(0.2)(x)
        if act == True:
            x = Activation('relu')(x)
        return x

    def conv_block(self,x,filters,kernel_size=(3,3),padding='same',strides=1):
        conv = self.bn_act(x)
        conv = Conv2D(filters,kernel_size,padding=padding,strides=strides)(conv)
        return conv
    def stem(self,x,filters,kernel_size=(3,3),padding='same',strides=1):
        conv = Conv2D(filters,kernel_size=kernel_size,padding=padding,strides=strides)(x)
        conv = self.conv_block(x,filters,kernel_size=kernel_size,padding=padding,strides=strides)

        shortcut = Conv2D(filters,kernel_size=(1,1),padding=padding,strides=strides)(x)
        shortcut = self.bn_act(shortcut,act=False)

        output = Add()([conv,shortcut])
        return output
    def residual_block(self,x,filters,kernel_size=(3,3),padding='same',strides=1):
        res = self.conv_block(x,filters,kernel_size=kernel_size,padding=padding,strides=strides)
        res = self.conv_block(res,filters,kernel_size=kernel_size,padding=padding,strides=1)

        shortcut = Conv2D(filters,kernel_size=(1,1),padding=padding,strides=strides)(x)
        shortcut = self.bn_act(shortcut,act=False)

        output = Add()([shortcut,res])
        return output
    def upsample_concat_block(self,x,xskip,filters):
        # I have changed to Conv2DTranspose from UpSampling due to deterministic strategy
        # u = UpSampling2D((2,2))(x)
        u = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)

        c = Concatenate()([u,xskip])
        # print(u.shape,xskip.shape,c.shape,"???")
        return c

    def get_model(self):
        f = [16,32,64,128,256]
        inputs = Input((self.IMG_HEIGHT,self.IMG_WIDTH,self.IMG_CHANNELS))

        # Encoder
        e0 = inputs
        e1 = self.stem(e0,f[0])
        e2 = self.residual_block(e1,f[1],strides=2)
        e3 = self.residual_block(e2,f[2],strides=2)
        e4 = self.residual_block(e3,f[3],strides=2)
        e5 = self.residual_block(e4,f[4],strides=2)

        # Bridge
        b0 = self.conv_block(e5,f[4],strides=1)
        b1 = self.conv_block(b0,f[4],strides=1)

        # Decoder
        u1 = self.upsample_concat_block(b1,e4,f[4])
        d1 = self.residual_block(u1,f[4])

        u2 = self.upsample_concat_block(d1,e3,f[3])
        d2 = self.residual_block(u2,f[3])

        u3 = self.upsample_concat_block(d2,e2,f[2])
        d3 = self.residual_block(u3,f[2])

        u4 = self.upsample_concat_block(d3,e1,f[1])
        d4 = self.residual_block(u4,f[1])
        
        outputs = Conv2D(1,(1,1),padding='same',activation='sigmoid')(d4)
        model = Model(inputs,outputs)
        return model
          



def U_NET(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS,kernel_initializer =  'he_uniform'):
    #Build the model for 128x128 and 2d images
    from keras.backend import tf as ktf

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    # s = Lambda(lambda x: x / 255)(inputs)
 
    #Contraction path
    c1 = Conv2D(64, (3, 3), activation='relu',  kernel_initializer=kernel_initializer, padding='same')(inputs)
    # c1 = Dropout(0.1)(c1)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(128, (3, 3), activation='relu',  kernel_initializer=kernel_initializer, padding='same')(p1)
    # c2 = Dropout(0.1)(c2)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p2)
    # c3 = Dropout(0.2)(c3)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(256, (3, 3), activation='relu',  kernel_initializer=kernel_initializer, padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p3)
    # c4 = Dropout(0.2)(c4)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p4)
    # c5 = Dropout(0.3)(c5)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u6)
    # c6 = Dropout(0.2)(c6)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c6)
     
    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u7)
    # c7 = Dropout(0.2)(c7)
    c7 = BatchNormalization()(c7)
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c7)
     
    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u8)
    # c8 = Dropout(0.1)(c8)
    c8 = BatchNormalization()(c8)
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c8)
     
    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u9)
    # c9 = Dropout(0.1)(c9)
    c9 = BatchNormalization()(c9)
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c9)
     
    outputs = Conv2D(IMG_CHANNELS, (1, 1), activation='sigmoid')(c9) # 'sigmoid'
     
    model = Model(inputs=[inputs], outputs=[outputs])
    return model



if __name__ == "__main__":
    # model = dense_unet_2d(256, 256, 1)
    # model.summary()
    # print(model.input_shape)
    # print(model.output_shape)

    # Test 2D if everything is working ok. 
    model = unet_2d_model_128_128(256, 256, 1)

    from keras_flops import get_flops

    model = build_resunet(256, 256, 32)
   
    model.summary()
    flops = get_flops(model, batch_size=1)
    print(f"FLOPS: {flops / 10 ** 9:.03} G")

    print(model.input_shape)
    print(model.output_shape)