################################################################################
#
# Define the network as described in Colorful Image Colorization by Zhang et al.
# in order to do accomplish the task of image colorization.
#
# The paper (see "../paper/Colorful_Image_Colorization___zhang_et_al.pdf")
# described the following network setup:
#
################################################################################

import pickle
import numpy as np

from keras import backend as K
from keras import Sequential
from keras.layers import Activation, Conv2D, BatchNormalization,\
    UpSampling2D, ZeroPadding2D
from keras.activations import relu, softmax
from keras import callbacks
from keras import optimizers

from .keras_util import DataGenerator, OutputProgress

from . import constants as c

################################################################################
# Custom loss functions


def multinomial_loss(y_true, y_pred):
    """
    :param y_pred: np.array, dimensions should be (n, h, w, q)
    :param soft_encoded: np.array, dimensions should be (n, h, w, q)
    Make sure all values are between 0 and 1, and that the sum of soft_encoded = 1
    :return: loss
    """

    loss = K.categorical_crossentropy(y_true, y_pred, axis=3) # Cross entropy
    loss = K.sum(loss, axis=1) # Sum over all width vectors
    loss = K.sum(loss, axis=1) # Sum over all height vectors
    loss = K.sum(loss) # Sum over all images in the batch

    return loss


def weighted_multinomial_loss(y_true, y_pred):
    """
    :param y_pred: np.array, dimensions should be (n, h, w, q)
    :param soft_encoded: np.array, dimensions should be (n, h, w, q)
    Make sure all values are between 0 and 1, and that the sum of soft_encoded = 1
    :return: loss
    """

    v = K.argmax(y_true, axis=3)
    v = K.one_hot(v, 262)
    v = v * c.weights
    v = K.sum(v, axis=3)

    loss = K.categorical_crossentropy(y_true, y_pred, axis=3)  # Cross entropy
    loss = K.dot(v, loss)
    loss = K.sum(loss, axis=1)  # Sum over all width vectors
    loss = K.sum(loss, axis=1)  # Sum over all height vectors
    loss = K.sum(loss)  # Sum over all images in the batch

    return loss


################################################################################
# Define the network as a Sequential Keras model

required_input_shape_ = (64, 64, 1)
required_output_shape = (64, 64, c.num_lab_bins)


def init_model(loss_function=multinomial_loss, batch_size=None):
    model = Sequential()

    # layer 1: (64x64x1) --> (32x32x64)
    model.add(ZeroPadding2D(input_shape=required_input_shape_, batch_size=batch_size, data_format='channels_last', name="layer1"))
    model.add(Conv2D(filters=64, kernel_size=3, padding="valid", strides=(1, 1)))
    model.add(Activation(relu))
    model.add(ZeroPadding2D())
    model.add(Conv2D(filters=64, kernel_size=3,  strides=(2, 2)))
    model.add(Activation(relu))
    model.add(BatchNormalization())

    # layer 2: (32x32x64) --> (16x16x128)
    model.add(ZeroPadding2D(name="layer2"))
    model.add(Conv2D(filters=128, kernel_size=3, padding="valid", strides=(1, 1)))
    model.add(Activation(relu))
    model.add(ZeroPadding2D())
    model.add(Conv2D(filters=128, kernel_size=3,  strides=(2, 2)))
    model.add(Activation(relu))
    model.add(BatchNormalization())

    # layer 3: (16x16x128) --> (8x8x256)
    model.add(ZeroPadding2D(name="layer3"))
    model.add(Conv2D(filters=256, kernel_size=3, padding="valid", strides=(1, 1)))
    model.add(Activation(relu))
    model.add(ZeroPadding2D())
    model.add(Conv2D(filters=256, kernel_size=3, padding="valid", strides=(1, 1)))
    model.add(Activation(relu))
    model.add(ZeroPadding2D())
    model.add(Conv2D(filters=256, kernel_size=3, strides=(2, 2)))
    model.add(Activation(relu))
    model.add(BatchNormalization())

    # layer 4:  (8x8x256) --> (8x8x512)
    model.add(ZeroPadding2D(name='layer4'))
    model.add(Conv2D(filters=512, kernel_size=3, padding="valid", strides=(1, 1)))
    model.add(Activation(relu))
    model.add(ZeroPadding2D())
    model.add(Conv2D(filters=512, kernel_size=3, padding="valid", strides=(1, 1)))
    model.add(Activation(relu))
    model.add(ZeroPadding2D())
    model.add(Conv2D(filters=512, kernel_size=3, padding="valid", strides=(1, 1)))
    model.add(Activation(relu))
    model.add(BatchNormalization())

    # layer 5: (8x8x512)--> (8x8x512)
    model.add(ZeroPadding2D(padding=(2, 2), name='layer5'))
    model.add(Conv2D(filters=512, kernel_size=3, padding="valid", strides=(1, 1), dilation_rate=(2, 2)))
    model.add(Activation(relu))
    model.add(ZeroPadding2D((2, 2)))
    model.add(Conv2D(filters=512, kernel_size=3, padding="valid", strides=(1, 1), dilation_rate=(2, 2)))
    model.add(Activation(relu))
    model.add(ZeroPadding2D((2, 2)))
    model.add(Conv2D(filters=512, kernel_size=3, padding="valid", strides=(1, 1), dilation_rate=(2, 2)))
    model.add(Activation(relu))
    model.add(BatchNormalization())

    # layer 6: (8x8x512)--> (8x8x512)
    model.add(ZeroPadding2D((2, 2), name='layer6'))
    model.add(Conv2D(filters=512, kernel_size=3, padding="valid", strides=(1, 1), dilation_rate=(2, 2)))
    model.add(Activation(relu))
    model.add(ZeroPadding2D((2, 2)))
    model.add(Conv2D(filters=512, kernel_size=3, padding="valid", strides=(1, 1), dilation_rate=(2, 2)))
    model.add(Activation(relu))
    model.add(ZeroPadding2D((2, 2)))
    model.add(Conv2D(filters=512, kernel_size=3, padding="valid", strides=(1, 1), dilation_rate=(2, 2)))
    model.add(Activation(relu))
    model.add(BatchNormalization())

    # layer 7: (8x8x512)--> (8x8x512)
    model.add(ZeroPadding2D(name='layer7'))
    model.add(Conv2D(filters=512, kernel_size=3, padding="valid", strides=(1, 1)))
    model.add(Activation(relu))
    model.add(ZeroPadding2D())
    model.add(Conv2D(filters=512, kernel_size=3, padding="valid", strides=(1, 1)))
    model.add(Activation(relu))
    model.add(ZeroPadding2D())
    model.add(Conv2D(filters=512, kernel_size=3, padding="valid", strides=(1, 1)))
    model.add(Activation(relu))
    model.add(BatchNormalization())

    # layer 8: (8x8x512)--> (16x15x256)
    model.add(UpSampling2D())
    model.add(ZeroPadding2D(name='layer8'))
    model.add(Conv2D(filters=256, kernel_size=3, strides=(1, 1)))
    model.add(Activation(relu))
    model.add(ZeroPadding2D())
    model.add(Conv2D(filters=256, kernel_size=3, padding="valid", strides=(1, 1)))
    model.add(Activation(relu))
    model.add(ZeroPadding2D())
    model.add(Conv2D(filters=256, kernel_size=3, padding="valid", strides=(1, 1)))
    model.add(Activation('relu'))

    # # layer 9: (16x15x256)--> (32x32x128)
    # model.add(UpSampling2D())
    # model.add(Conv2D(filters=128, kernel_size=3, padding="valid", strides=(1, 1)))
    # model.add(Activation(relu))
    # model.add(Conv2D(filters=128, kernel_size=3, padding="valid", strides=(1, 1)))
    # model.add(Activation(relu))
    # model.add(Conv2D(filters=128, kernel_size=3, padding="valid", strides=(1, 1)))
    # model.add(Activation(relu))
    # model.add(BatchNormalization())

    # output
    model.add(UpSampling2D((4, 4)))
    model.add(Conv2D(filters=c.num_lab_bins, kernel_size=1, strides=(1, 1)))
    model.add(Activation(softmax))

    adam = optimizers.adam()

    model.compile(loss=loss_function, optimizer=adam)

    return model

