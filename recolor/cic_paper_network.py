################################################################################
#
# Define the network as described in Colorful Image Colorization by Zhang et al.
# in order to do accomplish the task of image colorization.
#
# The paper (see "../paper/Colorful_Image_Colorization___zhang_et_al.pdf")
# described the following network setup:
#
################################################################################

import numpy as np
import pickle

from keras import Sequential
from keras.layers import Activation, Conv2D, BatchNormalization,\
    UpSampling2D, ZeroPadding2D
from keras.activations import relu, softmax
from keras import losses
from ImageProcessing import DataGenerator

################################################################################

required_input_shape_ = (224, 224, 1)


def init_model():
    model = Sequential()

    # layer 1:
    model.add(ZeroPadding2D(name="layer1"))
    model.add(Conv2D(filters=64, kernel_size=3, padding="valid", strides=(1, 1),
                     input_shape=required_input_shape_, data_format='channels_last'))
    model.add(Activation(relu))
    model.add(ZeroPadding2D())
    model.add(Conv2D(filters=64, kernel_size=3,  strides=(2, 2)))
    model.add(Activation(relu))
    model.add(BatchNormalization())

    # layer 2:
    model.add(ZeroPadding2D(name="layer2"))
    model.add(Conv2D(filters=128, kernel_size=3, padding="valid", strides=(1, 1)))
    model.add(Activation(relu))
    model.add(ZeroPadding2D())
    model.add(Conv2D(filters=128, kernel_size=3,  strides=(2, 2)))
    model.add(Activation(relu))
    model.add(BatchNormalization())

    # layer 3:
    model.add(ZeroPadding2D(name="layer3"))
    model.add(Conv2D(filters=256, kernel_size=3, padding="valid", strides=(1, 1)))
    model.add(Activation(relu))
    model.add(ZeroPadding2D())
    model.add(Conv2D(filters=256, kernel_size=3, padding="valid", strides=(1, 1)))
    model.add(Activation(relu))
    model.add(ZeroPadding2D())
    model.add(Conv2D(filters=256, kernel_size=3,  strides=(2, 2)))
    model.add(Activation(relu))
    model.add(BatchNormalization())

    # layer 4:
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

    # layer 5:
    model.add(ZeroPadding2D(padding=(2, 2), name='layer5'))
    model.add(Conv2D(filters=512, kernel_size=3, padding="valid", strides=(1, 1), dilation_rate=(2,2)))
    model.add(Activation(relu))
    model.add(ZeroPadding2D((2, 2)))
    model.add(Conv2D(filters=512, kernel_size=3, padding="valid", strides=(1, 1), dilation_rate=(2,2)))
    model.add(Activation(relu))
    model.add(ZeroPadding2D((2, 2)))
    model.add(Conv2D(filters=512, kernel_size=3, padding="valid", strides=(1, 1), dilation_rate=(2,2)))
    model.add(Activation(relu))
    model.add(BatchNormalization())

    # layer 6:
    model.add(ZeroPadding2D((2, 2), name='layer6'))
    model.add(Conv2D(filters=512, kernel_size=3, padding="valid", strides=(1, 1), dilation_rate=(2,2)))
    model.add(Activation(relu))
    model.add(ZeroPadding2D((2, 2)))
    model.add(Conv2D(filters=512, kernel_size=3, padding="valid", strides=(1, 1), dilation_rate=(2,2)))
    model.add(Activation(relu))
    model.add(ZeroPadding2D((2, 2)))
    model.add(Conv2D(filters=512, kernel_size=3, padding="valid", strides=(1, 1), dilation_rate=(2,2)))
    model.add(Activation(relu))
    model.add(BatchNormalization())

    # layer 7:
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

    # layer 8/7:
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


    """"# layer 8:
    model.add(UpSampling2D())
    model.add(Conv2D(filters=128, kernel_size=3, padding="valid", strides=(1, 1)))
    model.add(Activation(relu))
    model.add(Conv2D(filters=128, kernel_size=3, padding="valid", strides=(1, 1)))
    model.add(Activation(relu))
    model.add(Conv2D(filters=128, kernel_size=3, padding="valid", strides=(1, 1)))
    model.add(Activation(relu))
    model.add(BatchNormalization())"""

    # output
    model.add(Conv2D(filters=313, kernel_size=1, strides=(1, 1),
                     activation='softmax'
                     ))
    # model.add(softmax(313))
    # TODO add rescaling
    model.add(Conv2D(filters=2, kernel_size=1, padding="valid", strides=(1, 1)))
    model.add(UpSampling2D(size=(4, 4)))

    # model.build(required_input_shape_)
    model.compile(loss=losses.mean_squared_error, optimizer='adam')

    return model


def main():
    x = np.ones((1, 224, 224, 1))

    params = {'dim_in': (64 * 64),
              'dim_out': (64 * 64),
              'batch_size': 64,
              'n_channels_in': 1,
              'n_channels_out' : 3,
              'shuffle': True}

    with open('./train_ids.pickle', 'rb') as fp:
        train_partition = pickle.load(fp)

    with open('./validation_ids.pickle', 'rb') as fp:
        validation_partition = pickle.load(fp)

    labels = [] # Only important for the datagenerator model. label is generated at the time of loading

    training_generator = DataGenerator(train_partition, labels, **params)
    validation_generator = DataGenerator(validation_partition, labels, **params)

    model: Sequential = init_model()

# To use with model generator
    # model.fit_generator(generator=training_generator,
    #                     validation_data=validation_generator,
    #                     use_multiprocessing=True,
    #                     workers=6)

    y = model.predict(x, batch_size=1)

    print(y.shape) #"\n", y)

    model.summary()


if __name__ == '__main__':
    main()
