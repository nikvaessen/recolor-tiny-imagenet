import keras

from keras import Sequential
from keras.layers import ZeroPadding2D, Conv2D, BatchNormalization, UpSampling2D, Activation, Lambda
from keras.losses import mean_squared_error


def init_model():
    model: Sequential = Sequential()

    def i(x):
        return x

    model.add(Lambda(i, input_shape=(64, 64, 1)))

    # resolution stays equal with:
    #   * zero padding of (1,1)
    #   * kernel_size=3, stride=(1,1)
    #
    # resolution can be then 1/2 with stride=(2,2), or 1/3 by stride=(3,3), ect
    # e.g 64x64 becomes 32*32 with stride=(2,2) and 22*22 with stride=(3,3)
    model.add(ZeroPadding2D())
    model.add(Conv2D(filters=8, kernel_size=3, strides=(1, 1)))

    model.add(ZeroPadding2D())
    model.add(Conv2D(filters=200, kernel_size=3, strides=(1, 1)))

    model.add(UpSampling2D())
    model.add(ZeroPadding2D())
    model.add(Conv2D(filters=200, kernel_size=3, strides=(1, 1)))

    model.compile(optimizer="adam", loss=mean_squared_error)

    return model


def main():
    model = init_model()
    model.summary()

    idf = keras.backend.image_data_format()
    print(idf)


if __name__ == '__main__':
    main()
