################################################################################
#
# Define the network as described in Colorful Image Colorization by Zhang et al.
# in order to do accomplish the task of image colorization.
#
# The paper (see "../paper/Colorful_Image_Colorization___zhang_et_al.pdf")
# described the following network setup:
#
################################################################################

import time
from keras import backend as k
import pickle
import numpy as np
from keras import Sequential
from keras.layers import Activation, Conv2D, BatchNormalization,\
    UpSampling2D, ZeroPadding2D
from keras.activations import relu, softmax
from keras import losses
from keras import backend as K

from data_generator import DataGenerator
from image_logic import num_bins, probability_dist_to_ab_tensor, probability_dist_to_ab

import tensorflow as tf

# tf.enable_eager_execution()

################################################################################
# Custom loss functions


def multinomial_loss(predictions, soft_encodeds, weights):
    """
    :param predictions: np.array, dimensions should be (n, h, w, q)
    :param soft_encoded: np.array, dimensions should be (n, h, w, q)
    Make sure all values are between 0 and 1, and that the sum of soft_encoded = 1
    :return: loss
    """

    losses = 0
    for i in range(predictions.shape[0]):
        loss = 0
        for h in range(predictions.shape[1]):
            vs = np.array([weights[np.argmax(x)] for x in soft_encodeds[i, h]])
            loss = vs[:, np.newaxis] * np.dot(soft_encodeds[i, h],
                                              np.log(predictions[i, h] + 0.000000000000000001).transpose())
            loss = np.diag(loss)
            loss = - loss
            loss = np.sum(loss)
            losses += loss

    return losses


def multinomial_loss2(predictions, soft_encodeds):
    """
    :param predictions: np.array, dimensions should be (n, h, w, q)
    :param soft_encoded: np.array, dimensions should be (n, h, w, q)
    Make sure all values are between 0 and 1, and that the sum of soft_encoded = 1
    :return: loss
    """

    losses = 0
    for i in range(predictions.shape[0]):
        loss = 0
        for h in range(predictions.shape[1]):
            loss = np.dot(soft_encodeds[i, h],
                          np.log(predictions[i, h] + 0.000000000000000001).transpose())

            loss = np.diag(loss)
            loss = - loss
            loss = np.sum(loss)
            losses += loss

    return losses


def l2_loss(y_true, y_pred):
    y_pred = probability_dist_to_ab_tensor(y_pred)

    return losses.mean_squared_error(y_true, y_pred)


################################################################################
# Define the network as a Sequential Keras model

required_input_shape_ = (64, 64, 1)
required_output_shape = (64, 64, num_bins)
with open('../probabilities/weights.pickle', 'rb') as fp:
    weights = pickle.load(fp)



def init_model(loss_function=l2_loss, batch_size=None):
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

    # # layer 9: (8x8x512)--> (8x8x512)
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
    model.add(Conv2D(filters=num_bins, kernel_size=1, strides=(1, 1)))
    model.add(Activation(softmax))

    # TODO add rescaling?
    # model.add(Conv2D(filters=2, kernel_size=1, padding="valid", strides=(1, 1)))
    # model.add(UpSampling2D(size=(4, 4)))
    model.compile(loss=loss_function, optimizer='adam')

    return model


################################################################################
# Experimenting with running the model
def get_weights(bin, weights=weights):
    print('BIN', bin)
    return weights[bin]

def multinomial_loss(y_true, y_pred):
    print('Used')
    print('Prediction shape', y_pred.shape)
    """
    :param y_pred: np.array, dimensions should be (n, h, w, q)
    :param soft_encoded: np.array, dimensions should be (n, h, w, q)
    Make sure all values are between 0 and 1, and that the sum of soft_encoded = 1
    :return: loss
    """


    test0 = k.argmax(y_true, axis=3)
    test0 = k.cast(test0, k.floatx())
    # test0 = k.map_fn(get_weights, test0)

    test1 = k.categorical_crossentropy(y_true, y_pred, axis=3)

    test2 = k.dot(test0, test1)

    test3 = k.sum(test2, axis=1)

    test4 = k.sum(test3, axis=1)

    test5 = - test4

    test6 = k.sum(test5)

    return test6





    # losses = 0
    # for i in range(8):
    #     loss = 0
    #     for h in range(y_pred.shape[1]):
    #         vs = np.array([weights[np.argmax(x)] for x in y_true[i, h]])
    #         loss = vs[:, np.newaxis] * np.dot(y_true[i, h],
    #                                           np.log(y_pred[i, h] + 0.000000000000000001).transpose())
    #         loss = np.diag(loss)
    #         loss = - loss
    #         loss = np.sum(loss)
    #         losses += loss
    # print('Loss', losses)
    # return losses


def multinomial_loss2(predictions, soft_encodeds):
    """
    :param predictions: np.array, dimensions should be (n, h, w, q)
    :param soft_encoded: np.array, dimensions should be (n, h, w, q)
    Make sure all values are between 0 and 1, and that the sum of soft_encoded = 1
    :return: loss
    """

    losses = 0
    for i in range(predictions.shape[0]):
        loss = 0
        for h in range(predictions.shape[1]):
            loss = np.dot(soft_encodeds[i, h],
                                np.log(predictions[i, h] + 0.000000000000000001).transpose())

            loss = np.diag(loss)
            loss = - loss
            loss = np.sum(loss)
            losses += loss

    return losses
################################################################################
# Experimenting with running the model

def train_model_small_dataset():
    params = {
        'dim_in': (64, 64, 1),
        'dim_out': (64, 64, num_bins),
        'batch_size': 8,
        'shuffle': True,
        'mode': DataGenerator.mode_grey_in_ab_out
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

    model: Sequential = init_model(loss_function=l2_loss, batch_size=batch_size)
    # model.summary()

    # To use with model generator/
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=1,
                        verbose=1)

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


def get_gpu_info():
    print(K.tensorflow_backend._get_available_gpus())


def main():
    train_model_small_dataset()


if __name__ == '__main__':
    get_gpu_info()
    main()
