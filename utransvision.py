# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 14:22:09 2022
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import tensorflow as tf
import math
import segmentation_models as sm
from tensorflow.keras.applications import *


tfk = tf.keras
tfkl = tfk.layers
tfm = tf.math

class AddPositionEmbs(tfkl.Layer):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, trainable=True, **kwargs):
        super().__init__(trainable=trainable, **kwargs)
        self.trainable = trainable

    def build(self, input_shape):
        assert (
            len(input_shape) == 3
        ), f"Number of dimensions should be 3, got {len(input_shape)}"
        self.pe = tf.Variable(
            name="pos_embedding",
            initial_value=tf.random_normal_initializer(stddev=0.06)(
                shape=(1, input_shape[1], input_shape[2])
            ),
            dtype="float32",
            trainable=self.trainable,
        )
    def get_config(self):
        cfg = super().get_config()
        return cfg 

    def call(self, inputs):
        return inputs + tf.cast(self.pe, dtype=inputs.dtype)


class MultiHeadSelfAttention(tfkl.Layer):
    def __init__(self, *args, trainable=True, n_heads, **kwargs):
        super().__init__(trainable=trainable, *args, **kwargs)
        self.n_heads = n_heads
    
    def get_config(self):
        cfg = super().get_config()
        return cfg 

    def build(self, input_shape):
        hidden_size = input_shape[-1]
        n_heads = self.n_heads
        if hidden_size % n_heads != 0:
            raise ValueError(
                f"embedding dimension = {hidden_size} should be divisible by number of heads = {n_heads}"
            )
        self.hidden_size = hidden_size
        self.projection_dim = hidden_size // n_heads
        self.query_dense = tfkl.Dense(
            hidden_size, name=f"{self.name}/query")
        self.key_dense = tfkl.Dense(
            hidden_size, name=f"{self.name}/key")
        self.value_dense = tfkl.Dense(
            hidden_size, name=f"{self.name}/value")
        self.combine_heads = tfkl.Dense(
            hidden_size, name=f"{self.name}/out")

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], score.dtype)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):

        x = tf.reshape(
            x, (batch_size, -1, self.n_heads, self.projection_dim))

        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(
            attention, (batch_size,-1, self.hidden_size))
        output = self.combine_heads(concat_attention)
        return output, weights


class TransformerBlock(tfkl.Layer):
    """Implements a Transformer block."""

    def __init__(self, *args, n_heads, mlp_dim, dropout, trainable=True, **kwargs):
        super().__init__(*args, trainable=trainable, **kwargs)
        self.n_heads = n_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
    
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n_heads":8,"mlp_dim":512,"dropout":0.2})
        return cfg 
        
    def build(self, input_shape):
        self.att = MultiHeadSelfAttention(
            n_heads=self.n_heads,
            name="MultiHeadDotProductAttention_1",
        )
        self.mlpblock = tfk.Sequential(
            [
                tfkl.Dense(
                    self.mlp_dim,
                    activation="linear",
                    name=f"{self.name}/Dense_0"
                ),
                tfkl.Lambda(
                    lambda x: tfk.activations.gelu(x, approximate=False)
                )
                if hasattr(tfk.activations, "gelu")
                else tfkl.Lambda(
                    lambda x: tfa.activations.gelu(x, approximate=False)
                ),
                BatchNormalization(),
                tfkl.Dense(
                    input_shape[-1], name=f"{self.name}/Dense_1"),
                BatchNormalization()
            ],
            name="MlpBlock_3",
        )
        self.layernorm1 = tfkl.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_0"
        )
        self.layernorm2 = tfkl.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_2"
        )
        self.dropout = BatchNormalization()

    def call(self, inputs, training):
        x = self.layernorm1(inputs)

        x, weights = self.att(x)
        x = self.dropout(x, training=training)
        x = self.broadcast_add(x, inputs)
        y = self.layernorm2(x)
        y = self.mlpblock(y)
        return self.broadcast_add(x,y), weights
    
    def broadcast_add(self,x,y):
        return tf.math.add(x,y)
    


########################################
    
def UTranSvision(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS,kernel_initializer =  'he_uniform'):
    """Implements of the U-TranSvision architecture."""

    def transformer(y, filters, trainable=True):
        n_heads = 8
        n_layers = 1
        mlp_dim = 512
        dropout = 0.2
        hidden_size = filters
        
        y = tfkl.Reshape(
              (y.shape[1] * y.shape[2], hidden_size))(y)
        y = AddPositionEmbs(
              name=f"Transformer/posembed_input_{filters}", trainable=trainable)(y)

        # Transformer/Bridge or Decoder
        for n in range(n_layers):
            y, _ = TransformerBlock(
                n_heads=n_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                name=f"Transformer/bd_block_{n}_{filters}",
                trainable=trainable)(y)

        y = tfkl.LayerNormalization(epsilon=1e-6, name=f"Transformer/bd_norm_{filters}")(y)
        n_patch_sqrt = int(math.sqrt(y.shape[1]))
        
        y = tfkl.Reshape(target_shape=[n_patch_sqrt, n_patch_sqrt, hidden_size])(y)
        
        return y
    
    def get_loss(x,y):
        
        dice_loss  = sm.losses.DiceLoss()(x,y)                       # generate Dice Loss 
        focal_loss = sm.losses.CategoricalFocalLoss()(x,y)           # generate Categorical Focal Loss
        total_loss = dice_loss + (1 * focal_loss)                    # get total of the Dice and Categorical Focal Loss, 'binarycrossetropy'
        return total_loss

  
    def dummy_loss(y_true,y_pred):#,model):
        global L1,L2,L3
        
        # calculate losses
        loss0=get_loss(y_pred,y_true)
        loss1=get_loss(L1,y_true)
        loss2=get_loss(L2,y_true)    
        loss3=get_loss(L3,y_true) 
        
        # sum them up
        summe = tf.math.reduce_sum([loss0,loss1,loss2,loss3])
        
        return summe


    from keras.backend import tf as ktf
    global c6,c7,c8
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
 
    #Encoder
    c1 = Conv2D(16, (3, 3), activation='relu',  kernel_initializer=kernel_initializer, padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu',  kernel_initializer=kernel_initializer, padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(64, (3, 3), activation='relu',  kernel_initializer=kernel_initializer, padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    
    # Bridge
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c5)
    c5 = transformer(c5, 256)
    
    # Decoder 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c6)

    c6 = transformer(c6, 128)     
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u8)
    c8 = BatchNormalization()(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u9)
    c9 = BatchNormalization()(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c9)
     
    outputs = Conv2D(IMG_CHANNELS, (1, 1), activation='sigmoid',name="outputs")(c9)

    
    # DeepSupervision with summation of losses
    # global L1,L2,L3
    # L1 = Conv2DTranspose(1, (2, 2), strides=(8, 8), padding='same')(c6)#,name="L1_")(c6)
    # L2 = Conv2DTranspose(1, (2, 2), strides=(4, 4), padding='same')(c7)#,name="L2_")(c7)
    # L3 = Conv2DTranspose(1, (2, 2), strides=(2, 2), padding='same')(c8)#,name="L3_")(c8)
    # combined_loss = outputs
    # model = Model(inputs=[inputs],outputs=[combined_loss])    
    # return model, dummy_loss
    
    # USE BELOW SIDE FOR OUR APPROACH
    # DeepSupervision without summation of losses
    L1 = Conv2DTranspose(1, (2, 2), strides=(8, 8), padding='same',activation='sigmoid')(c6)#,name="L1_")(c6)
    L2 = Conv2DTranspose(1, (2, 2), strides=(4, 4), padding='same',activation='sigmoid')(c7)#,name="L2_")(c7)
    L3 = Conv2DTranspose(1, (2, 2), strides=(2, 2), padding='same',activation='sigmoid')(c8)#,name="L3_")(c8)



    model = Model(inputs=[inputs],outputs=[outputs,L1,L2,L3])


    return model





if __name__ == "__main__":
    from keras_flops import get_flops

    model = UTranSvision(256, 256, 1)
    model.summary()
    flops = get_flops(model, batch_size=1)
    print(f"FLOPS: {flops / 10 ** 9:.03} G")
    print(model.input_shape)
    print(model.output_shape)