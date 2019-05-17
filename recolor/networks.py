################################################################################
#
# Define multiple neural networks which can accomplish the recolorifcation task.
#
# Author(s): Nik Vaessen, Merline Sewina, Jade Cock
################################################################################

from keras import backend as K
from keras import Sequential
from keras import applications
from keras import optimizers

from keras.layers import Activation, Conv2D, BatchNormalization, UpSampling2D, ZeroPadding2D
from keras.activations import relu, softmax

if __name__ == '__main__' or __name__ == 'networks':
    import constants as c
else:
    from . import constants as c


################################################################################
# Custom loss functions

def get_multinomial_loss():
    return multinomial_loss


def multinomial_loss(y_true, y_pred):
    """
    :param y_pred: np.array, dimensions should be (n, h, w, q)
    :param soft_encoded: np.array, dimensions should be (n, h, w, q)
    Make sure all values are between 0 and 1, and that the sum of soft_encoded = 1
    :return: loss
    """

    loss = K.categorical_crossentropy(y_true, y_pred, axis=3)  # Cross entropy
    loss = K.sum(loss, axis=1)  # Sum over all width vectors
    loss = K.sum(loss, axis=1)  # Sum over all height vectors
    loss = K.sum(loss)  # Sum over all images in the batch

    return loss


def get_weighted_multinomial_loss(weights):
    def weighted_multinomial_loss(y_true, y_pred):
        """
        :param y_pred: np.array, dimensions should be (n, h, w, q)
        :param soft_encoded: np.array, dimensions should be (n, h, w, q)
        Make sure all values are between 0 and 1, and that the sum of soft_encoded = 1
        :return: loss
        """

        v = K.argmax(y_true, axis=3)
        v = K.one_hot(v, 262)
        v = v * weights
        v = K.sum(v, axis=3)

        loss = K.categorical_crossentropy(y_true, y_pred, axis=3)  # Cross entropy
        loss = K.dot(v, loss)
        loss = K.sum(loss, axis=1)  # Sum over all width vectors
        loss = K.sum(loss, axis=1)  # Sum over all height vectors
        loss = K.sum(loss)  # Sum over all images in the batch

        return loss

    return weighted_multinomial_loss


################################################################################
# Define the network as described in Colorful Image Colorization by Zhang et al.
# in order to do accomplish the task of image colorization.
#
# The paper (see "../paper/Colorful_Image_Colorization___zhang_et_al.pdf")
# describes the following network setup:


def init_cic_model(input_shape,
                   batch_size,
                   loss_function=multinomial_loss):
    model = Sequential()

    # layer 1: (64x64x1) --> (32x32x64)
    model.add(ZeroPadding2D(input_shape=input_shape, batch_size=batch_size,
                            data_format='channels_last', name="layer1"))
    model.add(Conv2D(filters=64, kernel_size=3, padding="valid", strides=(1, 1)))
    model.add(Activation(relu))
    model.add(ZeroPadding2D())
    model.add(Conv2D(filters=64, kernel_size=3, strides=(2, 2)))
    model.add(Activation(relu))
    model.add(BatchNormalization())

    # layer 2: (32x32x64) --> (16x16x128)
    model.add(ZeroPadding2D(name="layer2"))
    model.add(Conv2D(filters=128, kernel_size=3, padding="valid", strides=(1, 1)))
    model.add(Activation(relu))
    model.add(ZeroPadding2D())
    model.add(Conv2D(filters=128, kernel_size=3, strides=(2, 2)))
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

    # # layer 6: (8x8x512)--> (8x8x512)
    # model.add(ZeroPadding2D((2, 2), name='layer6'))
    # model.add(Conv2D(filters=512, kernel_size=3, padding="valid", strides=(1, 1), dilation_rate=(2, 2)))
    # model.add(Activation(relu))
    # model.add(ZeroPadding2D((2, 2)))
    # model.add(Conv2D(filters=512, kernel_size=3, padding="valid", strides=(1, 1), dilation_rate=(2, 2)))
    # model.add(Activation(relu))
    # model.add(ZeroPadding2D((2, 2)))
    # model.add(Conv2D(filters=512, kernel_size=3, padding="valid", strides=(1, 1), dilation_rate=(2, 2)))
    # model.add(Activation(relu))
    # model.add(BatchNormalization())
    #
    # # layer 7: (8x8x512)--> (8x8x512)
    # model.add(ZeroPadding2D(name='layer7'))
    # model.add(Conv2D(filters=512, kernel_size=3, padding="valid", strides=(1, 1)))
    # model.add(Activation(relu))
    # model.add(ZeroPadding2D())
    # model.add(Conv2D(filters=512, kernel_size=3, padding="valid", strides=(1, 1)))
    # model.add(Activation(relu))
    # model.add(ZeroPadding2D())
    # model.add(Conv2D(filters=512, kernel_size=3, padding="valid", strides=(1, 1)))
    # model.add(Activation(relu))
    # model.add(BatchNormalization())

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

    # output layer: (16x16x256)--> (64x64xn_bins)
    model.add(UpSampling2D((4, 4)))
    model.add(Conv2D(filters=c.num_lab_bins, kernel_size=1, strides=(1, 1)))
    model.add(Activation(softmax))

    opt = optimizers.adam()

    model.compile(loss=loss_function, optimizer=opt)

    return model

################################################################################
# A network which applies tranfser learning of VGG16


def init_vgg_transfer_model(input_shape,
                            batch_size,
                            loss_function=multinomial_loss,
                            ):
    vgg = applications.VGG16(weights='imagenet',
                             include_top=False,
                             input_shape=input_shape)

    # Freeze the vgg layers of the network
    for layer in vgg.layers:
        layer.trainable = False

    model = Sequential()
    model.add(vgg)

    # output layer: (2x2x512)--> (8x8x256)
    model.add(UpSampling2D((4, 4)))
    model.add(Conv2D(filters=256, kernel_size=1, strides=(1, 1)))
    model.add(Activation(relu))

    # output layer: (8x8x256)--> (32x32x256)
    model.add(UpSampling2D((4, 4)))
    model.add(Conv2D(filters=128, kernel_size=1, strides=(1, 1)))
    model.add(Activation(relu))

    # output layer: (16x16x256)--> (64x64xn_bins)
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(filters=c.num_lab_bins, kernel_size=1, strides=(1, 1)))
    model.add(Activation(softmax))

    adam = optimizers.adam()

    model.compile(loss=loss_function, optimizer=adam)

    return model

