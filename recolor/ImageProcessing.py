################################################################################
# Utility methods related to Data generator
# required to generate the data batches at training time
#
# author(s): Jade Cock
################################################################################

from skimage import io, color
import os
import numpy as np
import pickle
from image_logic import *

'''
x = cielba gray dimension
y = cielab image into the softencode function
'''

class DataGenerator:
    def __init__(self, list_IDs, batch_size=32, dim_in=(64 * 64), dim_out = (64 * 64), n_channels_in=1, n_channels_out=3,
             shuffle=True):
        '''
        :param list_IDs:
        :param batch_size: Size of the batches during training
        :param dim_in: Dimension of the input image (in Cielab)
        :param dim_out: Dimension of the output image (in Cielab)
        note : dim_in should be equal to dim_out
        :param n_channels_in: Number of colour channels - should be 1 (gray image)
        :param n_channels_out: Number of colour channels of the ouput image - should be 3 (colour image)
        :param shuffle: Whether to shuffle the input
        '''
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.shuffle = shuffle
        self.indexes = []
        self.on_epoch_end()

    def on_epoch_end(self):
        '''
        If shuffle is set to True: shuffles input at the beginning of each epoch
        :return:
        '''
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def read_image(self, image_name):
        '''
        Read an image and returns the rgb format, cielab and grayscale format
        :param image_name: image path

        :return: rgb format, and gray channel of cielab
        '''
        rgb = read_image(image_name)

        cielab = convert_rgb_to_lab(rgb)

        # gray_scale = np.zeros(cielab.shape) # uncomment to have an image convertible into rgb for plotting
        # gray_scale[:, :, 0] = cielab[:, :, 0]

        gray_channel = cielab[:, :, 0]

        return rgb, gray_channel

    # list_IDs_temp : Ids from the batch to be generated
    def __data_generation(self, list_IDs_temp, path):
        # Initialization
        X = np.empty((self.batch_size, *self.dim_in, self.n_channels_in))
        y = np.empty((self.batch_size, *self.dim_out, self.n_channels_out))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            pathImage = path + ID + '.JPEG'
            rgb, gray_channel = read_image(pathImage)
            X[i,] = rgb
            y[i,] = gray_channel # Change to rgb when changing the output

        return X, y

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

def test():
    path = 'C:/Users/jadec/My Tresors/Stockholm/KTH/Block 4/Deep learning/Project/DLIDS/Dataset/tiny-imagenet-200/train/n01443537/images/n01443537_3.JPEG'
    dg = DataGenerator([])
    rgb, cielab, gray = dg.read_image(path)


def get_ids():
    '''
    Create a list of the path to the images for the training set, validation set and test set
    '''

    # Training set
    rootdir = "../Dataset/tiny-imagenet-200/train"
    train_ids = []
    for subdirs, dirs, files in os.walk(rootdir):
        if len(files) == 500:
            for file in files:
                path = os.path.join(subdirs, file).replace('\\', '/')
                train_ids.append(path)
    with open('./train_ids.pickle', 'wb') as fp:
        pickle.dump(train_ids, fp)

    # validation set
    rootdir = "../Dataset/tiny-imagenet-200/val"
    validation_ids = []
    for subdirs, dirs, files in os.walk(rootdir):
        if len(files) != 0:
            for file in files:
                path = os.path.join(subdirs, file).replace('\\', '/')
                validation_ids.append(path)
    with open('./validation_ids.pickle', 'wb') as fp:
        pickle.dump(validation_ids, fp)

    # Test set
    rootdir = "../Dataset/tiny-imagenet-200/test"
    print("Debug")
    test_ids = []
    for subdirs, dirs, files in os.walk(rootdir):
        if files != 0:
            for file in files:
                path = os.path.join(subdirs, file).replace('\\', '/')
                test_ids.append(path)
    with open('./test_ids.pickle', 'wb') as fp:
        pickle.dump(test_ids, fp)



get_ids()



