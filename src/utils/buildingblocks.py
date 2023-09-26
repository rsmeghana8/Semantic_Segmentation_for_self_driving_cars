import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Dropout, MaxPooling2D, Conv2DTranspose


def conv_block(inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):

    layer = Conv2D(n_filters, kernel_size= 3, padding='same', kernel_initializer='he_normal')(inputs)
    layer = BatchNormalization(axis=3)(layer,training=True)
    layer = LeakyReLU()(layer)
    layer = Conv2D(n_filters, 3, padding='same', kernel_initializer='he_normal')(layer)
    layer = BatchNormalization(axis=3)(layer,training= False)
    layer = LeakyReLU()(layer)
    
    if dropout_prob > 0:
        layer = Dropout(dropout_prob)(layer)
        
    if max_pooling:
        next_layer = MaxPooling2D(pool_size=(2,2))(layer)
        
    else:
        next_layer = layer
        
    skip_connection = layer
    
    return next_layer, skip_connection

def upsampling_block(expansive_input, contractive_input,n_filters=32):

    up = Conv2DTranspose(n_filters, kernel_size = 3, strides=(2,2), padding='same')(expansive_input)
    merge = tf.concat([up,contractive_input],axis =3)


    layer = Conv2D(n_filters, kernel_size= 3, activation='relu', padding='same',
                 kernel_initializer= 'he_normal')(merge)
    layer = BatchNormalization(axis=3)(layer, training= False)
    layer = LeakyReLU() (layer)

    layer = Conv2D(n_filters, kernel_size= 3, activation='relu', padding='same',
                 kernel_initializer= 'he_normal')(layer)
    layer = BatchNormalization(axis=3)(layer, training= False)
    layer = LeakyReLU() (layer)

    return layer