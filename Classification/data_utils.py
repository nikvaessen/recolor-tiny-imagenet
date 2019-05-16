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
import image_logic
from keras import Sequential
from keras import callbacks
from keras import callbacks

################################################################################
# Data Generator class loaded at training time to provide the data in the batches


class DataGenerator(keras.utils.Sequence):
    def __init__(self, data_paths, batch_size, dim_in, dim_out, shuffle, mode, dataset_type):
        '''

        :param data_paths: paths to the data
        :param batch_size: size of the batches during trainng time
        :param dim_in: Dimensions of the input images
        :param dim_out: Dimensions of the output
        :param shuffle: Whether to shuffle the images in the different batches
        :param dataset_type: Whether it is for the validation or training set
        :return:
        '''

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.batch_size = batch_size
        self.leftovers = len(data_paths) % batch_size
        self.data_paths = data_paths
        self.shuffle = shuffle
        self.mode = mode

        if dataset_type == 'validation':
            self.keys = load_validation_keys()

        self.dataset_type = dataset_type

        with open('./label_id.pickle', 'rb') as fp:
            self.label_id = pickle.load(fp)

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
            if self.leftovers != 0:
                self.indices = self.indices[:-self.leftovers]
        else:
            self.indices = np.arange(len(self.data_paths) - self.leftovers)

        # print('Number of data', len(self.indices))

    def get_output(self, path):
        if self.dataset_type == 'training':
            label = path.split('/')[-1].split('_')[0]
        elif self.dataset_type == 'validation':
            label = path.split('/')[-1]
            label = self.keys[label]

        return self.label_id[label]

    def __data_generation(self, batch_paths):
        '''

        :param batch_paths: Paths of the images to be included in the batch
        :return:
        '''
        # Initialization

        X = np.empty((self.batch_size, *self.dim_in))
        y = np.empty((self.batch_size, self.dim_out))

        # Generate data
        for i, path in enumerate(batch_paths):
            if os.name == 'nt':
                path = path.replace('\\', '/')

            # Store sample
            # print(path)
            inp = image_logic.read_image(path)
            outp = self.get_output(path)
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


#######################################################################################
# Classes related to the labels
#

def name_to_float():
    '''
    Converts the name of the label into a float label
    :return: Creates dictionary from name to label and from label to name
    '''
    tiny_classes = [
        'n01443537', 'n01910747', 'n01917289', 'n01950731', 'n02074367', 'n09256479', 'n02321529',
        'n01855672', 'n02002724', 'n02056570', 'n02058221', 'n02085620', 'n02094433', 'n02099601', 'n02099712',
        'n02106662', 'n02113799', 'n02123045', 'n02123394', 'n02124075', 'n02125311', 'n02129165', 'n02132136',
        'n02480495', 'n02481823', 'n12267677', 'n01983481', 'n01984695', 'n02802426', 'n01641577'
    ]

    train_keys = load_keys()

    label_id = {}
    id_label = {}
    id_name = {}
    for i in range(len(tiny_classes)):
        label_id[tiny_classes[i]] = i
        id_label[i] = tiny_classes[i]
        id_name[i] = train_keys[tiny_classes[i]]
        print('Class', i, 'with label', tiny_classes[i], 'represents', train_keys[tiny_classes[i]])

    with open('./label_id.pickle', 'wb') as fp:
        pickle.dump(label_id, fp)

    with open('./id_label.pickle', 'wb') as fp:
        pickle.dump(id_label, fp)

    with open('./id_name.pickle', 'wb') as fp:
        pickle.dump(id_name, fp)


def load_keys():
    '''
    This function loads the file keeping track of the labels of the image
    key - > numerical value of the file (also the name of the folder they are in)
    value - > text value of the file
    '''
    label_path = "../data/tiny-imagenet-200/words.txt"
    keys = {}
    with open(label_path) as f:
        for line in f:
            key, val = line.split('\t')
            keys[key] = val
    return keys


def load_validation_keys():
    '''
    This function loads the file keeping track of the labels of the image
    key - > numerical value of the file (also the name of the folder they are in)
    value - > text value of the file
    '''
    label_path = "../data/tiny-imagenet-200/val/val_annotations.txt"
    keys = {}
    with open(label_path) as f:
        for line in f:
            key, val, _, _, _, _ = line.split('\t')
            keys[key] = val
    return keys

#######################################################################################
# Save images into their gray values
#


def save_gray_image(path, im_name):
    image = image_logic.read_image(path)
    new_path = '../data/vgg/' + im_name + '_gray.npz'
    gray = image_logic.rgb_to_gray(image)
    np.savez_compressed(new_path, gray=gray, colour=image)
    return new_path

# def save_input_output_ondisk():
#
#     # train_paths = []
#     # i = 0
#     # with open('./train_ids_tiny.pickle', 'rb') as fp:
#     #     train_ids = pickle.load(fp)
#     #     print('There are currently', len(train_ids), 'images in the training set')
#     #     for path in train_ids:
#     #         if i % 1000 == 0:
#     #             print('Saved ', i, 'documents')
#     #         namepath = 'train/' + path[49:-5]
#     #         new_path = save_input_output(path, namepath)
#     #         train_paths.append(new_path)
#     #         i += 1
#     #
#     # with open('../train_ids_npz.pickle', 'wb') as fp:
#     #     pickle.dump(train_paths, fp)
#     # print('Soft encoded training done')
#     #
#     validation_paths = []
#     i = 0
#     with open('./validation_ids_tiny.pickle', 'rb') as fp:
#         validation_ids = pickle.load(fp)
#         print('There are currently', len(validation_ids), 'images in the validation set')
#         for path in validation_ids:
#             if i % 1000 == 0:
#                 print('Saved', i, 'documents')
#             i += 1
#             namepath = 'val/' + path[37:-5]
#             # new_path = save_input_output(path, namepath)
#             new_path = '../data/npz-tiny-imagenet/' + namepath + '_intput_output.npz'
#             validation_ids.append(new_path)
#
#
#     with open('../validation_ids_npz.pickle', 'wb') as fp:
#         pickle.dump(validation_paths, fp)
#     # print('Soft encoded validation ids done!')
#
#     test_paths = []
#     i = 0
#     with open('./test_ids_tiny.pickle', 'rb') as fp:
#         test_ids = pickle.load(fp)
#         print('There are currently', len(test_ids), 'images in the test set')
#         for path in test_ids:
#             if i % 1000 == 0:
#                 print('Saved', i, 'documents')
#             i += 1
#             name_path = 'test/' + path[38:-5]
#             new_path = save_input_output(path, name_path)
#             test_paths.append(new_path)
#     #
#     with open('../test_ids_npz.pickle', 'wb') as fp:
#         pickle.dump(test_paths, fp)
#
#     print('Soft encoded test done!')


if __name__ == '__main__':
    name_to_float()



































