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

from data_generator import DataGenerator

import constants as c

################################################################################
# Custom loss functions

# with open('../probabilities/waitlist.pickle', 'rb') as fp:
#     weights = pickle.load(fp)
#     weights = K.variable(weights)


# def get_weights(bin, weights=weights):
#     print('BIN', bin)
#     return weights[bin]


def multinomial_loss(y_true, y_pred):
    """
    :param y_pred: np.array, dimensions should be (n, h, w, q)
    :param soft_encoded: np.array, dimensions should be (n, h, w, q)
    Make sure all values are between 0 and 1, and that the sum of soft_encoded = 1
    :return: loss
    """

    test1 = K.categorical_crossentropy(y_true, y_pred, axis=3)

    test2 = K.sum(test1, axis=1)

    test3 = K.sum(test2, axis=1)

    test5 = K.sum(test3)

    return test5


def weighted_multinomial_loss(y_true, y_pred):
    """
    :param y_pred: np.array, dimensions should be (n, h, w, q)
    :param soft_encoded: np.array, dimensions should be (n, h, w, q)
    Make sure all values are between 0 and 1, and that the sum of soft_encoded = 1
    :return: loss
    """

    print('Original shape', K.shape(y_true))
    test0 = K.max(y_true, axis=3)
    test0 = K.one_hot(test0, 262)
    print('Shape', K.shape(test0))
    test0 = K.cast(test0, K.floatx())
    # test0 = k.map_fn(get_weights, (test0))

    test1 = K.categorical_crossentropy(y_true, y_pred, axis=3)

    test2 = K.dot(test0, test1)

    test3 = K.sum(test2, axis=1)

    test4 = K.sum(test3, axis=1)

    test5 = - test4

    test6 = K.sum(test5)

    return test6


################################################################################
# Define the network as a Sequential Keras model

required_input_shape_ = (64, 64, 1)
required_output_shape = (64, 64, c.num_bins)


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
    model.add(Conv2D(filters=c.num_bins, kernel_size=1, strides=(1, 1)))
    model.add(Activation(softmax))

    adam = optimizers.adam()

    model.compile(loss=loss_function, optimizer=adam)

    return model

################################################################################
# Experimenting with running the model


def train_model_small_dataset_multinomial_loss():
    params = {
        'dim_in': (64, 64, 1),
        'dim_out': (64, 64, c.num_bins),
        'batch_size': 8,
        'shuffle': True,
        'mode': DataGenerator.mode_grey_in_softencode_out
    }

    with open('./train_ids.pickle', 'rb') as fp:
        train_partition = pickle.load(fp)

    with open('./validation_ids.pickle', 'rb') as fp:
        validation_partition = pickle.load(fp)

    # only use small amount of data :)
    batch_size = params['batch_size']
    train_partition = train_partition[0:4*batch_size]
    validation_partition = validation_partition[0:4*batch_size]

    print("using {} training samples".format(len(train_partition)))
    print("using {} validation samples".format(len(validation_partition)))

    training_generator = DataGenerator(train_partition, **params)
    validation_generator = DataGenerator(validation_partition, **params)

    model: Sequential = init_model(loss_function=multinomial_loss,
                                   batch_size=batch_size)
    # model.summary()

    tb_callback = callbacks.TensorBoard(log_dir='../tensorboard/test123',
                                        histogram_freq=0,
                                        write_graph=True,
                                        write_images=True)

    # # To use with model generator/
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=4,
                        verbose=1,
                        epochs=10,
                        callbacks=[tb_callback])

    # take a sample and try to predict
    import image_logic

    sample_rgb = image_logic.read_image(train_partition[18])
    sample_lab = image_logic.convert_rgb_to_lab(sample_rgb)

    sample_grey = sample_lab[:, :, 0:1]
    sample_grey = sample_grey.reshape(1, *sample_grey.shape)

    output = model.predict(sample_grey)

    ab = image_logic.probability_dist_to_ab(output)

    lab = np.zeros(sample_lab.shape)
    lab[:, :, 0:1] = sample_grey
    lab[:, :, 1:] = ab

    rgb = image_logic.convert_lab_to_rgb(lab)

    image_logic.plot_img_converted(sample_rgb, lambda x: rgb)


################################################################################

def main():
    train_model_small_dataset_multinomial_loss()


if __name__ == '__main__':
    main()
