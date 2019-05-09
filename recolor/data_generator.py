################################################################################
# Utility methods related to Data generator
# required to generate the data batches at training time
#
# author(s): Jade Cock, Nik Vaessen
################################################################################

import os
import pickle

import numpy as np
import keras

import image_logic


################################################################################
# Define different ways of reading the data

def load_image_grey_in_softencode_out(image_path):
    """
    Load an image, create the input and output of the network.

    The input is a gray-scale image as a (width*height*1) numpy array
    The output is a soft-encoding of the expected color bin as a
    (width*height*num_bins) numpy array

    :param image_path: the path to the image
    :return: tuple of x and y (input and output)
    """
    rgb = image_logic.read_image(image_path)
    cielab = image_logic.convert_rgb_to_lab(rgb)

    gray_channel = cielab[:, :, 0]
    gray_channel = gray_channel[:, :, np.newaxis]

    soft_encoding = image_logic.soft_encode_lab_img(cielab)

    return gray_channel, soft_encoding


def load_image_grey_in_ab_out(image_path):
    """
    Load an image, create the input and output of the network.

    The input is a gray-scale image as a (width*height*1) numpy array
    The output is a soft-encoding of the expected color bin as a
    (width*height*num_bins) numpy array

    :param image_path: the path to the image
    :return: tuple of x and y (input and output)
    """
    rgb = image_logic.read_image(image_path)
    cielab = image_logic.convert_rgb_to_lab(rgb)

    gray_channel = cielab[:, :, 0]
    gray_channel = gray_channel[:, :, np.newaxis]

    ab_channel = cielab[:, :, 1:]

    return gray_channel, ab_channel

################################################################################
# The DataGenerator class used to offload data i/o to CPU while GPU trains
# network


class DataGenerator(keras.utils.Sequence):
    mode_grey_in_ab_out = 'grey-in-ab-out'
    mode_grey_in_softencode_out = 'grey-in-softencode-out'

    modes = [mode_grey_in_ab_out, mode_grey_in_softencode_out]

    def __init__(self,
                 data_paths,
                 batch_size,
                 dim_in,
                 dim_out,
                 shuffle,
                 mode):
        '''
        :param data_paths: paths to the image files
        :param batch_size: Size of the batches during training
        :param dim_in: Dimension of the input image (in Cielab)
        :param dim_out: Dimension of the output image (in Cielab)
        :param shuffle: Whether to shuffle the input
        :param mode the mode of the data generator. Decides input and
        output of network
        '''
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.batch_size = batch_size
        self.data_paths = data_paths
        self.shuffle = shuffle

        if mode in DataGenerator.modes:
            if mode == DataGenerator.mode_grey_in_ab_out:
                self.image_load_fn = load_image_grey_in_ab_out
            elif mode == DataGenerator.mode_grey_in_softencode_out:
                self.image_load_fn = load_image_grey_in_softencode_out
        else:
            raise ValueError("expected mode to be one of", DataGenerator.modes)

        self.indexes = []

        self.on_epoch_end()

    def on_epoch_end(self):
        '''
        If shuffle is set to True: shuffles input at the beginning of each epoch
        :return:
        '''
        self.indexes = np.arange(len(self.data_paths))

        if self.shuffle:
            np.random.shuffle(self.indexes)

    # list_IDs_temp : Ids from the batch to be generated
    def __data_generation(self, batch_paths):
        # Initialization
        X = np.empty((self.batch_size, *self.dim_in))
        y = np.empty((self.batch_size, *self.dim_out))

        # Generate data
        for i, path in enumerate(batch_paths):
            # Store sample
            inp, outp = self.image_load_fn(path)
            X[i, ] = inp
            y[i, ] = outp

        return X, y

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        # Generate indexes of the batch
        indexes = self.indexes[
                  index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        batch_paths = [self.data_paths[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(batch_paths)

        return X, y

################################################################################
# stores paths to tiny-imagenet files as pickled arrays. These arrays
# are expected as input 'data_paths' in the DataGenerator above


def is_grey_image(fn):
    img = image_logic.read_image(fn)

    return img.shape == (64, 64)


def generate_data_paths_and_pickle():
    '''
    Create a list of the path to the images for the training set, validation set and test set
    '''

    image_extension = ".JPEG"

    # Training set
    rootdir = "../data/tiny-imagenet-200/train"
    train_ids = []

    for subdirs, dirs, files in os.walk(rootdir):
        for file in files:
            path = os.path.join(subdirs, file).replace('\\', '/')
            if os.path.splitext(file)[1] == image_extension and \
                    not is_grey_image(path):
                train_ids.append(path)

    with open('./train_ids.pickle', 'wb') as fp:
        pickle.dump(train_ids, fp)

    print("created training id's")

    # validation set
    rootdir = "../data/tiny-imagenet-200/val"
    validation_ids = []
    for subdirs, dirs, files in os.walk(rootdir):
        for file in files:
            path = os.path.join(subdirs, file).replace('\\', '/')
            if os.path.splitext(file)[1] == image_extension and \
                    not is_grey_image(path):
                validation_ids.append(path)

    with open('./validation_ids.pickle', 'wb') as fp:
        pickle.dump(validation_ids, fp)

    print("created validation id's")

    # Test set
    rootdir = "../data/tiny-imagenet-200/test"

    test_ids = []
    for subdirs, dirs, files in os.walk(rootdir):
        for file in files:
            path = os.path.join(subdirs, file).replace('\\', '/')
            if os.path.splitext(file)[1] == image_extension and \
                    not is_grey_image(path):
                test_ids.append(path)

    with open('./test_ids.pickle', 'wb') as fp:
        pickle.dump(test_ids, fp)

    print("created test id's")

################################################################################
# Create pickled files when this file is run directly


if __name__ == '__main__':
    generate_data_paths_and_pickle()



