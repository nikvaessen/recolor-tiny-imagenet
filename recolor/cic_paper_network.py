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
import tensorflow as tf # Can comment out when not in test mode


from keras_util import DataGenerator, OutputProgress

import constants as c

################################################################################
# Custom loss functions

with open('../probabilities/waitlist2.pickle', 'rb') as fp:
    weights = pickle.load(fp)
weights = np.array(weights)


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
    v = v * weights
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
# methods helpful for using the model


def train_model_small_dataset_multinomial_loss():
    # define sensible default parameters
    params = {
        'dim_in': (64, 64, 1),
        'dim_out': (64, 64, c.num_bins),
        'batch_size': 8,
        'shuffle': True,
        'mode': DataGenerator.mode_grey_in_softencode_out
    }

    with open('./train_ids_tiny.pickle', 'rb') as fp:
        train_partition = pickle.load(fp)

    with open('./validation_ids_tiny.pickle', 'rb') as fp:
        validation_partition = pickle.load(fp)
    # define data generators
    train_partition = c.training_set_file_paths
    validation_partition = c.validation_set_file_paths

    train_partition = train_partition[0:4*8]
    validation_partition = validation_partition[0:4*8]

    training_generator = DataGenerator(train_partition, **params)
    validation_generator = DataGenerator(validation_partition, **params)

    model: Sequential = init_model(loss_function=weighted_multinomial_loss,
                                   batch_size=batch_size)
    # model.summary()
    model: Sequential = init_model(loss_function=multinomial_loss,
                                   batch_size=params['batch_size'])

    tb_callback = callbacks.TensorBoard(log_dir='../tensorboard',
                                        histogram_freq=5,
                                        write_graph=True,
                                        write_images=True)

    lr_callback = callbacks.ReduceLROnPlateau()

    op_callback = OutputProgress(train_partition[5:8], required_input_shape_,
                                 "../tensorboard/")

    save_callback = callbacks.ModelCheckpoint("weights.{epoch:02d}-{val_loss:.2f}.hdf5", period=2)

    # To use with model generator/
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=4,
                        verbose=1,
                        epochs=10,
                        callbacks=[tb_callback, lr_callback, op_callback, save_callback])

    # take a sample and try to predict
    import image_util as image_logic

    import image_logic
    # Image 1
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

    ## Image 2

    sample_rgb = image_logic.read_image(train_partition[30])
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

def test():
    # v = K.argmax(y_true, axis=3)
    # v = K.one_hot(v, 262)
    #
    # weights = K.variable(weight_list)
    # test = weights * v
    # print('Test', test.shape)
    # weights = K.expand_dims(weights, axis=0)
    # weights = K.repeat_elements(weights, rep=64, axis=0)
    # print('update')
    #
    # print('update')
    # print(weights.shape)
    # v = K.dot(v, weights)

    ############################""
    a = np.array([[[5, 2, 6],
                    [7, 7, 3],
                    [0, 0, 0],
                    [6, 6, 5]],

                   [[2, 2, 3],
                    [1, 3, 3],
                    [8, 1, 9],
                    [0, 5, 7]]])
    weights = np.array([2, 0, 1])
    a = K.constant(a)
    b = K.argmax(a, axis=2)
    c = K.one_hot(b, 3)
    d = c * weights
    e = K.sum(d, axis=2)
    sess = tf.Session()
    print(sess.run(a))
    print('*_' * 10)
    print(sess.run(b))
    print('*_' * 10)
    print(sess.run(c))
    print('*_' * 10)
    print(sess.run(d))
    print('*_' * 10)
    print(sess.run(e))





if __name__ == '__main__':
    main()
    # test()
