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


'''
x = cielba gray dimension
y = cielab image into the softencode function
'''


def load_image(image_path):
    """
    Load an image, create the input and output of the network.

    The input is a gray-scale image as a (width*height) numpy array
    The output is a soft-encoding of the expected color bin as a
    (width*height*num_bins) numpy array

    :param image_path: the path to the image
    :return: tuple of x and y (input and output)
    """
    rgb = image_logic.read_image(image_path)
    cielab = image_logic.convert_rgb_to_lab(rgb)

    gray_channel = cielab[:, :, 0]
    soft_encoding = image_logic.soft_encode_lab_img(cielab)

    return gray_channel, soft_encoding


class DataGenerator(keras.utils.Sequence):

    def __init__(self,
                 data_paths,
                 batch_size,
                 dim_in,
                 dim_out,
                 shuffle):
        '''
        :param data_paths: paths to the image files
        :param batch_size: Size of the batches during training
        :param dim_in: Dimension of the input image (in Cielab)
        :param dim_out: Dimension of the output image (in Cielab)
        :param shuffle: Whether to shuffle the input
        '''
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.batch_size = batch_size
        self.data_paths = data_paths
        self.shuffle = shuffle

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
            gray_channel, soft_encoding = load_image(path)
            X[i, ] = gray_channel
            y[i, ] = soft_encoding

        return X, y

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        batch_paths = [self.data_paths[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(batch_paths)

        return X, y


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


if __name__ == '__main__':
    generate_data_paths_and_pickle()



