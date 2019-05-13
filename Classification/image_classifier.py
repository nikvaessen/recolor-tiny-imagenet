################################################################################
#
# Implementation of the VGG classifier as it is in the original layer, whilst
# changing the input format and modifying the classifier layers
#
# authors = Jade Cock
################################################################################
import data_utils
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras import callbacks
import pickle
import os
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

################################################################################
# enable training of network

subfolders = ['models', 'tensorboard-log-dir', 'progression']
def create_result_dir(path):
    name = 'vgg-classification'

    experiment_path = './' + path + '/' + name + '/'

    if not os.path.isdir(experiment_path):
        os.mkdir(experiment_path)

    index = 0
    while True:
        if index > 100:
            print("please do not run so many experiments on the same machine!")
            exit()

        experiment_path_subfolder = os.path.join(experiment_path, "run{}".format(index))

        if os.path.isdir(experiment_path_subfolder):
            index += 1
        else:
            os.mkdir(experiment_path_subfolder)
            for sf in subfolders:
                os.mkdir(os.path.join(experiment_path_subfolder, sf))
            return experiment_path_subfolder


def train(model: Sequential, mode):

    if mode == 'gray':
        with open('./train_ids_gray.pickle', 'rb') as fp:
            training_paths = pickle.load(fp)

        with open('./validation_ids_gray.pickle', 'rb') as fp:
            validation_paths = pickle.load(fp)

    elif mode == 'recoloured':
        with open('./train_ids_recolour.pickle', 'rb') as fp:
            training_paths = pickle.load(fp)

        with open('./validation_ids_recolour.pickle', 'rb') as fp:
            validation_paths = pickle.load(fp)

    ### params
    dim_in = (64, 64, 1)
    dim_out = (64, 64, 3)
    shuffle = True
    batch_size = 32


    training_generator = data_utils.DataGenerator(training_paths, batch_size, dim_in, dim_out, shuffle, mode)
    validation_generator = data_utils.DataGenerator(validation_paths, batch_size, dim_in, dim_out, shuffle, mode)

    callback_list = list()

    print("using tensorboard")
    tb_callback = callbacks.TensorBoard(log_dir=create_result_dir('tensorboard'))
    callback_list.append(tb_callback)

    saving_period = 10
    print("saving model every {} epochs".format(saving_period))
    p_save_callback = callbacks.ModelCheckpoint(create_result_dir('models'),
                                                period=saving_period)
    callback_list.append(p_save_callback)

    print("saving best model")
    best_save_callback = callbacks.ModelCheckpoint(create_result_dir(create_result_dir('bestmodel')),
                                                   save_best_only=True)
    callback_list.append(best_save_callback)

    n_workers = 2
    n_epochs = 100
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=n_workers,
                        verbose=1,
                        epochs=n_epochs,
                        callbacks=callback_list)















































