################################################################################
# Utility methods related to keras functionality:
# Data generator required to generate the data batches at training time
# Callback for saving progress of image coloring
#
# author(s): Jade Cock, Nik Vaessen
################################################################################

import os
import pickle

import numpy as np
import keras

if os.name == "nt":
    import image_util
    import constants as c
else:
    from . import image_util
    from . import constants as c

################################################################################
# Define different ways of reading the data

def load_compressed_files(image_path):
    """
    Load the compressed file image_path.npz where the attribute input
    is the cielab image, and where the attribute ouput is the soft
    encoding version of the expected colour bin as a
    (width*height*num_bins) np array
    :param image_path: path to the npz file
    :return: the cielab image and the soft encoded image in that order
    """

    compressed = np.load(image_path)
    cielab, soft_encode = compressed['input'], compressed['output']

    return cielab[:, :, 0:1], soft_encode


def load_image_grey_in_softencode_out(image_path):
    """
    Load an image, create the input and output of the network.

    The input is a gray-scale image as a (width*height*1) numpy array
    The output is a soft-encoding of the expected color bin as a
    (width*height*num_bins) numpy array

    :param image_path: the path to the image
    :return: tuple of x and y (input and output)
    """
    filename = os.path.split(image_path)[-1]
    filename = os.path.splitext(filename)[0]
    se_filename = os.path.join(c.soft_encoding_training_and_val_dir,
                               filename + c.soft_encoding_filename_postfix)

    rgb = image_util.read_image(image_path)
    cielab = image_util.convert_rgb_to_lab(rgb)

    gray_channel = cielab[:, :, 0]
    gray_channel = gray_channel[:, :, np.newaxis]

    if os.path.exists(se_filename):
        soft_encoding = np.load(se_filename)['arr_0']
    else:
        soft_encoding = image_util.soft_encode_lab_img(cielab)

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
    rgb = image_util.read_image(image_path)
    cielab = image_util.convert_rgb_to_lab(rgb)

    gray_channel = cielab[:, :, 0]
    gray_channel = gray_channel[:, :, np.newaxis]

    ab_channel = cielab[:, :, 1:]

    return gray_channel, ab_channel


################################################################################
# The DataGenerator class used to offload data i/o to CPU while GPU trains
# network


class DataGenerator(keras.utils.Sequence):
    compressed_mode = 'compressed-mode'
    mode_grey_in_ab_out = 'grey-in-ab-out'
    mode_grey_in_softencode_out = 'grey-in-softencode-out'

    modes = [compressed_mode, mode_grey_in_ab_out, mode_grey_in_softencode_out]

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
        self.leftovers = c.n_training_set_tiny % batch_size
        self.data_paths = data_paths
        self.shuffle = shuffle

        if mode in DataGenerator.modes:
            if mode == DataGenerator.compressed_mode:
                self.image_load_fn = load_compressed_files
            elif mode == DataGenerator.mode_grey_in_ab_out:
                self.image_load_fn = load_image_grey_in_ab_out
            elif mode == DataGenerator.mode_grey_in_softencode_out:
                self.image_load_fn = load_image_grey_in_softencode_out
        else:
            raise ValueError("expected mode to be one of", DataGenerator.modes)

        self.indices = []

        self.on_epoch_end()

    def on_epoch_end(self):
        '''
        If shuffle is set to True: shuffles input at the beginning of each epoch
        :return:
        '''

        if self.shuffle:
            self.indices = np.arange(len(self.data_paths))
            np.random.shuffle(self.indices)
            self.indices = self.indices[:-self.leftovers]
        else:
            self.indices = np.arange(len(self.data_paths) - self.leftovers)

    # list_IDs_temp : Ids from the batch to be generated
    def __data_generation(self, batch_paths):
        # Initialization
        X = np.empty((self.batch_size, *self.dim_in))
        y = np.empty((self.batch_size, *self.dim_out))

        # Generate data
        for i, path in enumerate(batch_paths):
            if os.name == 'nt':
                path = path.replace('\\', '/')  # Activate for Windows

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


################################################################################
# stores paths to tiny-imagenet files as pickled arrays. These arrays
# are expected as input 'data_paths' in the DataGenerator above


def is_grey_image(fn):
    img = image_util.read_image(fn)
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
# Define custom callback(s) for our use-case


class OutputProgress(keras.callbacks.Callback):

    def __init__(self,
                 image_paths_file,
                 input_shape,
                 root_dir,
                 must_convert_pdist=True,
                 every_n_epochs=5):
        super().__init__()

        with open(os.path.abspath(image_paths_file), 'r') as f:
            image_paths = f.readlines()

        image_paths = [path.strip() for path in image_paths]

        self.batch = np.empty((len(image_paths), *input_shape))

        for idx, path in enumerate(image_paths):
            rgb = image_util.read_image(path)
            lab = image_util.convert_rgb_to_lab(rgb)
            grey = lab[:, :, 0:1]
            self.batch[idx, ] = grey

        self.root_dir = root_dir
        self.period = every_n_epochs
        self.epochs_since_last_save = 0
        self.must_convert_pdist=must_convert_pdist

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1

        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            self.save_images(str(epoch + 1))

    def on_train_end(self, logs=None):
        self.save_images('training_end')

    def save_images(self, epoch_string: str):
        y = self.model.predict(self.batch)

        if self.must_convert_pdist:
            y = image_util.probability_dist_to_ab(y)

        for idx in range(self.batch.shape[0]):
            l = self.batch[idx, ]
            ab = y[idx, ]

            lab = np.empty((l.shape[0], l.shape[1], 3))

            lab[:, :, 0:] = l
            lab[:, :, 1:] = ab

            rgb = (image_util.convert_lab_to_rgb(lab) * 255).astype(np.uint8)
            path = os.path.join(self.root_dir,
                                "img_{}_epoch_{}.png".format(idx, epoch_string))
            image_util.save_image(path, rgb)


################################################################################
# Create soft_encode
# Aiming to accelerate training
#

lab_bin_centers = c.lab_bin_centers


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


def get_available_classes():
    file_counter = 0
    labels = load_keys()
    train_path = "../data/tiny-imagenet-200/train"
    counter_gray = 0

    for subdirs, dirs, files in os.walk(train_path):
        if len(files) == 500:
            file_counter += 1

            label = files[0][:9]
            label_name = labels[label]
            print(file_counter, ': ', label_name, '->', label)


def get_tinytiny_dataset():
    tiny_classes = [
        'n01443537', 'n01910747', 'n01917289', 'n01950731', 'n02074367', 'n09256479', 'n02321529',
        'n01855672', 'n02002724', 'n02056570', 'n02058221', 'n02085620', 'n02094433', 'n02099601', 'n02099712',
        'n02106662', 'n02113799', 'n02123045', 'n02123394', 'n02124075', 'n02125311', 'n02129165', 'n02132136',
        'n02480495', 'n02481823', 'n12267677', 'n01983481', 'n01984695', 'n02802426', 'n01641577'
    ]

    image_extension = ".JPEG"

    # Training set
    rootdir = "../data/tiny-imagenet-200/train"
    train_ids = []
    for subdirs, dirs, files in os.walk(rootdir):
        if len(files) == 500 and files[0][:9] in tiny_classes:
            for file in files:
                path = os.path.join(subdirs, file).replace('\\', '/')
                if os.path.splitext(file)[1] == image_extension and \
                        not is_grey_image(path):
                    train_ids.append(path)
    with open('./train_ids_tiny.pickle', 'wb') as fp:
        pickle.dump(train_ids, fp)

    print("created training id's")

    # # validation set
    rootdir = "../data/tiny-imagenet-200/val"
    valkeys = load_validation_keys()
    validation_ids = []
    print('test')
    for subdirs, dirs, files in os.walk(rootdir):
        for file in files:
            if os.path.splitext(file)[1] == image_extension and valkeys[file] in tiny_classes:
                # print(file)
                path = os.path.join(subdirs, file).replace('\\', '/')
                if os.path.splitext(file)[1] == image_extension and \
                        not is_grey_image(path):
                    validation_ids.append(path)

    with open('./validation_ids_tiny.pickle', 'wb') as fp:
        pickle.dump(validation_ids, fp)

    print("created validation id's")
    #
    # # Test set -> Isn't annotated should still do ?
    rootdir = "../data/tiny-imagenet-200/test"

    test_ids = []
    for subdirs, dirs, files in os.walk(rootdir):
        for file in files:
            path = os.path.join(subdirs, file).replace('\\', '/')
            if os.path.splitext(file)[1] == image_extension and \
                    not is_grey_image(path):
                test_ids.append(path)
    #
    with open('./test_ids_tiny.pickle', 'wb') as fp:
        pickle.dump(test_ids, fp)
    #
    print("created test id's")

def save_soft_encode(path):
    image = image_util.read_image(path)
    lab = image_util.convert_rgb_to_lab(image)
    se = image_util.soft_encode_lab_img(lab)
    new_path = '../data/soft_encoded/' + path[-16:-5] + '_soft_encoded.npz'
    np.savez_compressed(new_path, se)
    return new_path


def save_softencode_ondisk():


    # with open('./done_train_ids_tiny.pickle', 'rb') as fp:
    #     done = pickle.load(fp)
    #
    # with open('../train_ids_soft_encoded.pickle', 'rb') as fp:
    #     train_paths = pickle.load(fp)
    # i = 0
    # with open('./train_ids_tiny.pickle', 'rb') as fp:
    #     train_ids = pickle.load(fp)
    #     print('There are currently', len(train_ids), 'images in the training set')
        # for path in train_ids:
        #     if i % 1000 == 0:
        #         print('Saved ', i, 'documents')
        #     namepath = path[49:-5]
        #     if namepath not in done:
        #         new_path = save_soft_encode(path, namepath)
        #         train_paths.append(new_path)
        #         i += 1

    # with open('../train_ids_soft_encoded.pickle', 'wb') as fp:
    #     pickle.dump(train_paths, fp)
    # print('Soft encoded training done')

    validation_paths = []
    # i = 0
    # with open('./validation_ids_tiny.pickle', 'rb') as fp:
    #     validation_ids = pickle.load(fp)
    #     print('There are currently', len(validation_ids), 'images in the validation set')
        # for path in validation_ids:
        #     if i % 1000 == 0:
        #         print('Saved', i, 'documents')
        #     i+=1
        #     namepath = path[37:-5]
        #     new_path = save_soft_encode(path, namepath)
        #     validation_ids.append(new_path)

    # with open('../validation_ids_soft_encoded.pickle', 'wb') as fp:
    #     pickle.dump(validation_paths, fp)
    # print('Soft encoded validation ids done!')
    #
    test_paths = []
    i = 0
    with open('./test_ids_tiny.pickle', 'rb') as fp:
        test_ids = pickle.load(fp)
        print('There are currently', len(test_ids), 'images in the test set')
        for path in test_ids:
            if i % 1000 == 0:
                print('Saved', i, 'documents')
            i += 1
            new_path = save_soft_encode(path)
            test_paths.append(new_path)

    with open('../test_ids_soft_encoded.pickle', 'wb') as fp:
        pickle.dump(test_paths, fp)

    print('Soft encoded test done!')

def count_number_of_images():

    with open('../train_ids_soft_encoded.pickle', 'rb') as fp:
        train_paths = pickle.load(fp)
    print("Number of training images: ", len(train_paths))

    with open('../validation_ids_soft_encoded.pickle', 'rb') as fp:
        validation_paths = pickle.load(fp)
    print("Number of validation images: ", len(validation_paths))

    with open('../test_ids_soft_encoded.pickle', 'rb') as fp:
        test_paths = pickle.load(fp)
    print("Number of training images: ", len(test_paths))

################################################################################
# Create compress objects for inputs and outputs
#

def save_input_output(path, newpath):
    image = image_util.read_image(path)
    lab = image_util.convert_rgb_to_lab(image)
    se = image_util.soft_encode_lab_img(lab)

    new_path = '../data/npz-tiny-imagenet/' + newpath + '_intput_output.npz'
    np.savez_compressed(new_path, input=lab, output=se)
    return new_path


def save_input_output_ondisk():

    # train_paths = []
    # i = 0
    # with open('./train_ids_tiny.pickle', 'rb') as fp:
    #     train_ids = pickle.load(fp)
    #     print('There are currently', len(train_ids), 'images in the training set')
    #     for path in train_ids:
    #         if i % 1000 == 0:
    #             print('Saved ', i, 'documents')
    #         namepath = 'train/' + path[49:-5]
    #         new_path = save_input_output(path, namepath)
    #         train_paths.append(new_path)
    #         i += 1
    #
    # with open('../train_ids_npz.pickle', 'wb') as fp:
    #     pickle.dump(train_paths, fp)
    # print('Soft encoded training done')
    #
    validation_paths = []
    i = 0
    with open('./validation_ids_tiny.pickle', 'rb') as fp:
        validation_ids = pickle.load(fp)
        print('There are currently', len(validation_ids), 'images in the validation set')
        for path in validation_ids:
            if i % 1000 == 0:
                print('Saved', i, 'documents')
            i += 1
            namepath = 'val/' + path[37:-5]
            # new_path = save_input_output(path, namepath)
            new_path = '../data/npz-tiny-imagenet/' + namepath + '_intput_output.npz'
            validation_ids.append(new_path)


    with open('../validation_ids_npz.pickle', 'wb') as fp:
        pickle.dump(validation_paths, fp)
    # print('Soft encoded validation ids done!')

    test_paths = []
    i = 0
    with open('./test_ids_tiny.pickle', 'rb') as fp:
        test_ids = pickle.load(fp)
        print('There are currently', len(test_ids), 'images in the test set')
        for path in test_ids:
            if i % 1000 == 0:
                print('Saved', i, 'documents')
            i += 1
            name_path = 'test/' + path[38:-5]
            new_path = save_input_output(path, name_path)
            test_paths.append(new_path)
    #
    with open('../test_ids_npz.pickle', 'wb') as fp:
        pickle.dump(test_paths, fp)

    print('Soft encoded test done!')

###############################################################################
# Test loadgin functions

def test_loading():

    cielab, se = load_compressed_files('../data/npz-tiny-imagenet/train/n01641577_0_intput_output.npz')

    print('Cielab', cielab.shape)
    print('Soft encoding', se.shape)

    cielab = image_util.convert_lab_to_rgb(cielab)
    image_util.plot_image(cielab)





################################################################################
# Create pickled files for used by DataGenerator when this file is run directly



if __name__ == '__main__':
    # generate_data_paths_and_pickle()
    # get_available_classes()
    # get_tinytiny_dataset()
    # save_softencode_ondisk()
    # check_already_encoded_images()
    # save_input_output_ondisk()
    test_loading()






