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


class DataGenerator:
    def __init__(self, list_IDs, batch_size=32, dim_in=(64 * 64, 1), dim_out = (64 * 64), n_channels_in=1, n_channels_out=3,
             shuffle=True):
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
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def read_image(self, image_name):
        rgb = io.imread(image_name)
        cielab = color.rgb2lab(rgb)
        grayScale = cielab[0]  # Check it is the gray channel

        return rgb, cielab, grayScale

    # list_IDs_temp : Ids from the batch to be generated
    def __data_generation(self, list_IDs_temp, path):
        # Initialization
        X = np.empty((self.batch_size, *self.dim_in, self.n_channels_in))
        y = np.empty((self.batch_size, *self.dim_out, self.n_channels_out))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            pathImage = path + ID + '.JPEG'
            rgb, cielab, grayScale = read_image(pathImage)
            X[i,] = grayScale
            y[i,] = cielab # Change to rgb when changing the output

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


def get_ids():
    rootdir = "../Dataset/tiny-imagenet-200/train"
    print("Debug")
    train_ids = []
    for subdirs, dirs, files in os.walk(rootdir):
        sub = subdirs.replace("\\", "/")
        if sub[-6:] == 'images':
            for subdirs2, dirs2, files2 in os.walk(sub):
                for file2 in files2:
                    if file2[0:2] != '._':
                        path = rootdir + sub + '/' + file2
                        train_ids.append(path)
    with open('./train_ids.pickle', 'wb') as fp:
        pickle.dump(train_ids, fp)

    rootdir = "../Dataset/tiny-imagenet-200/test/images"
    print("Debug")
    test_ids = []
    for subdirs, dirs, files in os.walk(rootdir):
        for file in files:
            if file[0:2] != '._':
                path = rootdir + file
                test_ids.append(path)
    with open('./test_ids.pickle', 'wb') as fp:
        pickle.dump(test_ids, fp)

    rootdir = "../Dataset/tiny-imagenet-200/val"
    print("Debug")
    validation_ids = []
    for subdirs, dirs, files in os.walk(rootdir):
        sub = subdirs.replace("\\", "/")
        if sub[-6:] == 'images':
            for subdirs2, dirs2, files2 in os.walk(sub):
                for file2 in files2:
                    if file2[0:2] != '._':
                        path = rootdir + sub + '/' + file2
                        validation_ids.append(path)

    with open('./validation_ids.pickle', 'wb') as fp:
        pickle.dump(validation_ids, fp)





