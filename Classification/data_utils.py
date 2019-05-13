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

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        # Generate indices of the batch
        indices = self.indices[
                  index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        batch_paths = [self.data_paths[k] for k in indices]

        # Generate data
        X, y = self.__data_generation(batch_paths)

        return X, y




































