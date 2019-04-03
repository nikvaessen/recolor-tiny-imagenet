################################################################################
#
# Define the network as described in Colorful Image Colorization by Zhang et al.
# in order to do accomplish the task of image colorization.
#
# The paper (see "../paper/Colorful_Image_Colorization___zhang_et_al.pdf")
# described the following network setup:
#
################################################################################

from keras import Sequential
from keras.layers import Activation, Conv2D, BatchNormalization
from keras.activations import relu

################################################################################


def init_model():
    model = Sequential()

    # layer 1:
    model.add(Conv2D())
    model.add(Activation(relu))
    model.add(Conv2D())
    model.add(Activation(relu))
    model.add(BatchNormalization())

    # layer 2:
    model.add(Conv2D())
    model.add(Activation(relu))
    model.add(Conv2D())
    model.add(Activation(relu))
    model.add(BatchNormalization())

    # layer 3:
    model.add(Conv2D())
    model.add(Activation(relu))
    model.add(Conv2D())
    model.add(Activation(relu))
    model.add(Conv2D())
    model.add(Activation(relu))
    model.add(BatchNormalization())

    # layer 4:
    model.add(Conv2D())
    model.add(Activation(relu))
    model.add(Conv2D())
    model.add(Activation(relu))
    model.add(Conv2D())
    model.add(Activation(relu))
    model.add(BatchNormalization())

    # layer 5:
    model.add(Conv2D())
    model.add(Activation(relu))
    model.add(Conv2D())
    model.add(Activation(relu))
    model.add(Conv2D())
    model.add(Activation(relu))
    model.add(BatchNormalization())

    # layer 6:
    model.add(Conv2D())
    model.add(Activation(relu))
    model.add(Conv2D())
    model.add(Activation(relu))
    model.add(Conv2D())
    model.add(Activation(relu))
    model.add(BatchNormalization())

    # layer 7:
    model.add(Conv2D())
    model.add(Activation(relu))
    model.add(Conv2D())
    model.add(Activation(relu))
    model.add(Conv2D())
    model.add(Activation(relu))
    model.add(BatchNormalization())

    # layer 8:
    model.add(Conv2D())
    model.add(Activation(relu))
    model.add(Conv2D())
    model.add(Activation(relu))
    model.add(Conv2D())
    model.add(Activation(relu))
    model.add(BatchNormalization())
