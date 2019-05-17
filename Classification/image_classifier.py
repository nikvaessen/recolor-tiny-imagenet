################################################################################
#
# Implementation of the VGG classifier as it is in the original layer, whilst
# changing the input format and modifying the classifier layers
#
# authors = Jade Cock
################################################################################

import data_utils
from keras import applications
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
from keras import callbacks
import pickle
import os

###############################################################################
# Classifier definition

image_shape = (64, 64, 3)


def init_model():

    # Pre-trained VGG16
    model = applications.VGG16(weights=None, include_top=False, input_shape=image_shape)

    # Personalised Classifier
    o = model.output
    o = Flatten()(o)
    o = Dense(4096, activation='relu')(o)
    o = Dropout(0.5)(o)
    o = Dense(4096, activation='relu')(o)
    predictions = Dense(30, activation='softmax')(o)

    # Finalised model
    opt = optimizers.SGD(nesterov=True, model=0.9)
    final_model = Model(input=model.input, output=predictions)
    final_model.compile(loss='categorical_crossentropy',
                        optimizer=opt,
                        metrics=['accuracy'])

    return final_model

################################################################################
# enable training of network


subfolders = ['models/', 'tensorboard/', 'bestmodels/']


def create_result_dir(experiment_path):

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
            if os.name == 'nt':
                experiment_path_subfolder = experiment_path_subfolder.replace('\\', '/')
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

    elif mode == 'colour':
        with open('./train_ids_tiny.pickle', 'rb') as fp:
            training_paths = pickle.load(fp)
        with open('./validation_ids_tiny.pickle', 'rb') as fp:
            validation_paths = pickle.load(fp)

    # params
    dim_in = (64, 64, 3)
    shuffle = True
    batch_size = 64
    n_classes = 30
    dim_out = (n_classes)

    training_generator = data_utils.DataGenerator(training_paths, batch_size, dim_in, dim_out, shuffle,
                                                  mode, 'training')
    validation_generator = data_utils.DataGenerator(validation_paths, batch_size, dim_in, dim_out, shuffle,
                                                    mode, 'validation')

    callback_list = list()

    directory = create_result_dir('./vgg_classification')
    print('Directory:', directory)

    print("using tensorboard")
    tensor_directory = directory + '/' + 'tensorboard'
    tb_callback = callbacks.TensorBoard(log_dir=tensor_directory)
    callback_list.append(tb_callback)

    saving_period = 3
    print("saving model every {} epochs".format(saving_period))
    model_dir = os.path.join(directory, 'models', 'model.{epoch:02d}-loss_{val_loss:.2f}.hdf5')
    p_save_callback = callbacks.ModelCheckpoint(model_dir,
                                                period=saving_period)
    callback_list.append(p_save_callback)

    print("saving best model")
    bestmodels_dir = directory + '/' + 'bestmodels/best_model.hdf5'
    best_save_callback = callbacks.ModelCheckpoint(bestmodels_dir,
                                                   save_best_only=True,
                                                   mode='val_acc')
    callback_list.append(best_save_callback)

    lr_reduce = callbacks.ReduceLROnPlateau(monitor='val_acc',
                                            factor=0.2,
                                            patience=4
                                            )
    callback_list.append(lr_reduce)

    n_workers = 2
    n_epochs = 30

    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=n_workers,
                        verbose=1,
                        epochs=n_epochs,
                        callbacks=callback_list)


def main():
    model = init_model()
    train(model, 'colour')


if __name__ == '__main__':
    main()
















































