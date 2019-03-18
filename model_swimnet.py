"""
# ----------------------------------------------------------------------------------------
# The model of SWiM-Net.

# Title: Snapshot Wide-field Multispectral Imaging behind Scattering Medium using Convolutional Neural Networks

# Author: Hua Zhang, Liangcai Cao, Xiaohan Li, Chengyu Wang, Jiache Wu, Michael Gehm, Dacid J. Brady, Guofan jin

# Institution: Hololab,Department of Precision Instruments, Beijing, China, 100084

# Date: 02.28.2019

# ---------------------------------------------------------------------------------------
"""

from __future__ import print_function
from keras.layers import Input, Conv2D, Dropout, Activation, MaxPooling2D, UpSampling2D, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.models import Model

# define swim-net
def defined_model_swimnet_hololab():
    inputs = Input((128, 128, 1))
    print("inputs:", inputs.shape)

    conv1 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    db1 = DB(x=conv1)
    print("Contracting Layer1:", db1.shape)

    conv2 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(MaxPooling2D(pool_size=(2, 2))(db1))
    db2 = DB(x=conv2)
    print("Contracting Layer2:", db2.shape)

    conv3 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(MaxPooling2D(pool_size=(2, 2))(db2))
    db3 = DB(x=conv3)
    print("Contracting Layer3:", db3.shape)

    conv4 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(MaxPooling2D(pool_size=(2, 2))(db3))
    db4 = DB(x=conv4)
    print("Contracting Layer4:", db4.shape)

    conv5 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(MaxPooling2D(pool_size=(2, 2))(db4))
    db5 = DB(x=conv5)
    print("Contracting Layer5:", db5.shape)

    conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(MaxPooling2D(pool_size=(2, 2))(db5))
    db6 = DB(x=conv6)
    print("Contracting Layer6:", db6.shape)

    conv7 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(MaxPooling2D(pool_size=(2, 2))(db6))
    db7 = DB(x=conv7)
    print("Contracting Layer7:", db7.shape)

    conv8 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(MaxPooling2D(pool_size=(2, 2))(db7))
    db8 = DB(x=conv8)
    up8 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(db8))
    print("Expending Layer 8:", db8.shape)

    conv9 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Concatenate(axis=3)([db7, up8]))
    db9 = DB(x=conv9)
    up9 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(db9))
    print("Expending Layer 9:", db9.shape)

    conv10 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Concatenate(axis=3)([db6, up9]))
    db10 = DB(x=conv10)
    up10 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(db10))
    print("Expending Layer 10:", db10.shape)

    conv11 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Concatenate(axis=3)([db5, up10]))
    db11 = DB(x=conv11)
    up11 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(db11))
    print("Expending Layer 11:", db11.shape)

    conv12 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Concatenate(axis=3)([db4, up11]))
    db12 = DB(x=conv12)
    up12 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(db12))
    print("Expending Layer 12:", db12.shape)

    conv13 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Concatenate(axis=3)([db3, up12]))
    db13 = DB(x=conv13)
    up13 = Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(db13))
    print("Expending Layer 13:", db13.shape)

    conv14 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Concatenate(axis=3)([db2, up13]))
    db14 = DB(x=conv14)
    up14 = Conv2D(8, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(db14))
    print("Expending Layer 14:", db14.shape)

    conv15 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Concatenate(axis=3)([db1, up14]))
    db15 = DB(x=conv15)
    print("Expending Layer 15:", db15.shape)

    conv_16 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(db15)
    conv_17 = Conv2D(3, 1, activation='sigmoid')(conv_16)
    print("Final output:", conv_17.shape)

    model = Model(inputs=inputs, outputs=conv_17)

    return model


# the function of dense block
def DB(x, concat_axis=3, nb_layers=3, growth_rate=8,dropout_rate=0.05,weight_decay=1E-4):
    list_feat = [x]
    for i in range(nb_layers):
        x = BatchNormalization(axis=concat_axis,gamma_regularizer=l2(weight_decay),beta_regularizer=l2(weight_decay))(x)
        x = Activation('relu')(x)
        x = Conv2D(growth_rate, (5, 5), dilation_rate=(2, 2),kernel_initializer="he_uniform",padding="same",kernel_regularizer=l2(weight_decay))(x)
        if dropout_rate:
            x = Dropout(dropout_rate)(x)

        list_feat.append(x)
        x = Concatenate(axis=concat_axis)(list_feat)

    return x