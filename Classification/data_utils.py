################################################################################
# Utility methods related to keras functionality:
# Data generator required to generate the data batches at training time
#
# Utility methods related to data creation :
# Create the images that will be trained
#
# author(s): Jade Cock
################################################################################

import os
import pickle

import numpy as np
import keras
from keras import Sequential
from keras import callbacks

################################################################################
# Data Generator class loaded at training time to provide the data in the batches

class DataGenerator(keras.utils.Sequence):
    def __init__(self, data_paths, batch_size, dim_in, dim_out, shuffle, mode):
        '''

        :param data_paths: paths to the data
        :param batch_size: size of the batches during trainng time
        :param dim_in: Dimensions of the input images
        :param dim_out: Dimensions of the output
        :param shuffle: Whether to shuffle the images in the different batches
        :return:
        '''

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.batch_size = batch_size
        self.leftovers = len(data_paths) - batch_size
        self.data_paths = data_paths
        self.shuffle = shuffle
        self.mode = mode

        self.indices = []
        self.on_epoch_end()



    def on_epoch_end(self):
        '''
        If shuffle is set to True, the dataset is shuffled after each epochs
        :return:
        '''

        if self.shuffle:
            self.indices = np.arange(len(self.data_paths))
            np.random.shuffle(self.indices)
            self.indices = self.indices[:-self.leftovers]
        else:
            self.indices = np.arange(len(self.data_paths) - self.leftovers)

    def __data_generation(self, batch_paths):
        '''

        :param batch_paths: Paths of the images to be included in the batch
        :return:
        '''
        # Initialization

        X = np.empty((self.batch_size, *self.dim_in))
        y = np.empty((self.batch_size, *self.dim_out))

        # Generate data
        for i, path in enumerate(batch_paths):
            if os.name == 'nt':
                path = path.replace('\\', '/')

            # Store sample
            # print(path)
            inp, outp = self.image_load_fn(path)
            X[i, ] = inp
            y[i, ] = outp

        return X, y

    def get_generators(self):
        params = {
            'dim_in': self.dim_in,
            'dim_out': self.dim_out,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'mode': self.mode
        }

        if self.mode == 'gray':
            with open('./train_ids_gray.pickle', 'rb') as fp:
                training_paths = pickle.load(fp)

            with open('./validation_ids_gray.pickle', 'rb') as fp:
                validation_paths = pickle.load(fp)

        elif self.mode == 'recoloured':
            with open('./train_ids_recolour.pickle', 'rb') as fp:
                training_paths = pickle.load(fp)

            with open('./validation_ids_recolour.pickle', 'rb') as fp:
                validation_paths = pickle.load(fp)

        print('using dataset:', self.dataset)
        print("using {} training samples".format(len(training_paths)))
        print('using {} validation samples'.format(len(validation_paths)))

        training_generator = DataGenerator(training_paths, **params)
        validation_generator = DataGenerator(validation_paths, **params)

        return training_generator, validation_generator

    def get_init_model(self):
        model = self.init_model()
        return model

def train(model: Sequential, mode):

    training_generator, validation_generator = DataGenerator.get_generators(mode)
    callback_list = list()

    print("using tensorboard")
    tb_callback = callbacks.TensorBoard(log_dir=config.tensorboard_log_dir)
    callback_list.append(tb_callback)

    print("saving model every {} epcohs".format(config.periodically_save_model_period))
    p_save_callback = callbacks.ModelCheckpoint(config.periodically_save_model_path,
                                                period=config.periodically_save_model_period)
    callback_list.append(p_save_callback)

    if config.save_best_model:
        print("saving best model")
        best_save_callback = callbacks.ModelCheckpoint(config.save_best_model_path,
                                                       save_best_only=True)
        callback_list.append(best_save_callback)

    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=config.n_workers,
                        verbose=1,
                        epochs=config.n_epochs,
                        callbacks=callback_list)







































