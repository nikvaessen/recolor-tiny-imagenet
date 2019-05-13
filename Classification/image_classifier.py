################################################################################
#
# Implementation of the VGG classifier as it is in the original layer, whilst
# changing the input format and modifying the classifier layers
#
# authors = Jade Cock
################################################################################

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

###############################################################################
# Classifier definition

image_shape = (64, 64, 3)

def init_model():

    # Pre-trained VGG16
    model = applications.VGG16(weights='imagenet', include_top=False, input_shape=image_shape)

    # Freeze the first layers of the network
    for layer in model.layers[:5]:
        layer.trainable = False

    # Personalised Classifier
    o = model.output
    o = Flatten()(o)
    o = Dense(4096, activation='relu')(o)
    o = Dropout(0.5)(o)
    o = Dense(4096, activation='relu')(o)
    predictions = Dense(30, activation='softmax')(o)

    # Finalised model
    final_model = Model(input=model.input, output=predictions)
    final_model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=0.0001, momentum=0.85), metrics=['accuracy'])








































