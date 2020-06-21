from __future__ import print_function

import cv2
import numpy as np
from keras import backend as K
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, concatenate
from keras.models import Model
from keras.optimizers import Adam, Adadelta

from constants import img_rows, img_cols

K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

#Override Dropout. Make it able at test time.
def call(self, inputs, training=None):
    if 0. < self.rate < 1.:
        noise_shape = self._get_noise_shape(inputs)
        def dropped_inputs():
            return K.dropout(inputs, self.rate, noise_shape,
                             seed=self.seed)
        if (training):
            return K.in_train_phase(dropped_inputs, inputs, training=training)
        else:
            return K.in_test_phase(dropped_inputs, inputs, training=None)
    return inputs

Dropout.call = call

def get_unet(dropout):
    inputs = Input((1, img_rows, img_cols))
    conv1 = Conv2D(32, (3, 3), activation='relu', border_mode='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)


    conv2 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)


    conv3 = Conv2D(128, (3, 3), activation='relu', border_mode='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', border_mode='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', border_mode='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', border_mode='same')(conv5)

    if dropout:
        conv5 = Dropout(0.5)(conv5)
        #conv5 = Dropout(0.5)

    #up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=1)

    conv6 = Conv2D(256, (3, 3), activation='relu', border_mode='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', border_mode='same')(conv6)

    #up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)

    conv7 = Conv2D(128, (3, 3), activation='relu', border_mode='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', border_mode='same')(conv7)

    #up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)

    conv8 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(conv8)

    #up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=1)

    conv9 = Conv2D(32, (3, 3), activation='relu', border_mode='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', border_mode='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def get_unet_multi(dropout,channels,n_class):
    inputs = Input((img_rows, img_cols, channels))
    conv1 = Conv2D(32, (3, 3), activation='relu', border_mode='same',data_format='channels_last')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', border_mode='same',data_format='channels_last')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2),data_format='channels_last')(conv1)


    conv2 = Conv2D(64, (3, 3), activation='relu', border_mode='same',data_format='channels_last')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', border_mode='same',data_format='channels_last')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2),data_format='channels_last')(conv2)


    conv3 = Conv2D(128, (3, 3), activation='relu', border_mode='same',data_format='channels_last')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', border_mode='same',data_format='channels_last')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2),data_format='channels_last')(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', border_mode='same',data_format='channels_last')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', border_mode='same',data_format='channels_last')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2),data_format='channels_last')(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', border_mode='same',data_format='channels_last')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', border_mode='same',data_format='channels_last')(conv5)

    if dropout:
        conv5 = Dropout(0.5)(conv5)
        #conv5 = Dropout(0.5)

    #up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    up6 = concatenate([UpSampling2D(size=(2, 2),data_format='channels_last')(conv5), conv4], axis=3)

    conv6 = Conv2D(256, (3, 3), activation='relu', border_mode='same',data_format='channels_last')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', border_mode='same',data_format='channels_last')(conv6)

    #up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    up7 = concatenate([UpSampling2D(size=(2, 2),data_format='channels_last')(conv6), conv3], axis=3)

    conv7 = Conv2D(128, (3, 3), activation='relu', border_mode='same',data_format='channels_last')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', border_mode='same',data_format='channels_last')(conv7)

    #up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    up8 = concatenate([UpSampling2D(size=(2, 2),data_format='channels_last')(conv7), conv2], axis=3)

    conv8 = Conv2D(64, (3, 3), activation='relu', border_mode='same',data_format='channels_last')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', border_mode='same',data_format='channels_last')(conv8)

    #up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    up9 = concatenate([UpSampling2D(size=(2, 2),data_format='channels_last')(conv8), conv1], axis=3)

    conv9 = Conv2D(32, (3, 3), activation='relu', border_mode='same',data_format='channels_last')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', border_mode='same',data_format='channels_last')(conv9)

    conv10 = Conv2D(n_class, (1, 1), activation='softmax',data_format='channels_last')(conv9)

    model = Model(input=inputs, output=conv10)

    print(model.summary())

    # model.compile(optimizer=Adam(lr=8e-6), loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=Adadelta(), loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    return model
