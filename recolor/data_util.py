import pickle
import os

import numpy as np

if __name__ == '__main__' or __name__ == 'keras_util':
    import image_util
    import constants as c
else:
    from . import image_util
    from . import constants as c


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
    from keras_util import load_compressed_files

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
