
from src.utils.buildingblocks import conv_block, upsampling_block
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Dropout, MaxPooling2D, Conv2DTranspose
from src.config.configuration import PrepareModelConfig

class PrepareModel:
    def __init__(self, config: PrepareModelConfig):
        self.config = config
        self.input_size = tuple(self.config.input_size)
        self.n_classes = self.config.n_classes



    def unet_model(self, n_filters= 32):
        input_size = self.input_size
        n_classes = self.n_classes
     

        # Encoder
        
        inputs = tf.keras.Input(input_size)
    
        cblock1 = conv_block(inputs, n_filters)
        cblock2 = conv_block(cblock1[0], n_filters* 2)
        cblock3 = conv_block(cblock2[0], n_filters * 4)
        cblock4 = conv_block(cblock3[0], n_filters * 8, dropout_prob=0.3)
        cblock5 = conv_block(cblock4[0], n_filters * 16, dropout_prob=0.3, max_pooling=False)
    

        # Decoder
    
        ublock6 = upsampling_block(cblock5[0], cblock4[1], n_filters *8)
        ublock7 = upsampling_block(ublock6, cblock3[1], n_filters *4)
        ublock8 = upsampling_block(ublock7, cblock2[1], n_filters * 2)
        ublock9 = upsampling_block(ublock8, cblock1[1], n_filters)
    
        layer = Conv2D(n_filters, 3, padding='same', kernel_initializer='he_normal')(ublock9)
        layer = BatchNormalization(axis =3)(layer, training= True)
        layer = LeakyReLU()(layer)
    
        layer = Conv2D(n_classes, 1, padding='same')(layer)
        layer = BatchNormalization(axis =3)(layer, training=False)
        layer = LeakyReLU()(layer)
    
        model = tf.keras.Model(inputs = inputs, outputs= layer)
        return model