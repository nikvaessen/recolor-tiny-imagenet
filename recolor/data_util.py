import pickle
import os

import numpy as np

if __name__ == '__main__' or __name__ == 'keras_util':
    import image_util
    import constants as c
else:
    from . import image_util
    from . import constants as c

import tensorflow as tf


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

    with open('./saved_objects/train_ids.pickle', 'wb') as fp:
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

    with open('./saved_objects/validation_ids.pickle', 'wb') as fp:
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

    with open('./saved_objects/test_ids.pickle', 'wb') as fp:
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
# Create compressed tensor record data
#

def create_id_labels():
    '''
    Create files to map labels to their name labels, and to an arbitraty id from 0 to 200
    :return:
    '''
    keys = load_keys()
    label_path = "../data/tiny-imagenet-200/wnids.txt"
    labels = []
    with open(label_path) as f:
        for line in f:
            labels.append(line.split('\n')[0])

    id_label = {}
    label_id = {}
    label_name = {}

    for i, label in enumerate(labels):
        name = keys[label]
        id_label[i] = label
        label_id[label] = i
        label_name[label] = name
        print('Label', label, 'coressponds to', name, 'with class', i)


def wrap_int64(value):
    '''
    :param value:
    :return: returns a list of number into a tfrecord object
    '''
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def wrap_bytes(value):
    '''

    :param value:
    :return: returns a list of object into a tfrecord object
    '''
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert(image_paths, out_path):
    '''
    This functions creates tf record object.
    Inputs are the paths to the rgb pictures. This functions saves
    the different images in their cielab format and in their soft encoded format
    :param image_path: Path to the images to convert into a tfrecord object
    :param out_path: Path where to save the tfrecord object
    :return:
    '''

    print("Converting: " + out_path)
    # Number of images. Used when printing the progress.
    num_images = len(image_paths)

    # Open a TFRecordWriter for the output-file.
    with tf.python_io.TFRecordWriter(out_path) as writer:
        # Iterate over all the image-paths
        for i, (path, label) in enumerate(image_paths):
            # Read the images
            rgb = np.array(image_util.read_image(path))
            cie = np.array(image_util.convert_rgb_to_lab(rgb))
            se = np.array(image_util.soft_encode_lab_img(cie))

            # Convert them into raw bytes
            cie_bytes = cie.tostring()
            se_bytes = se.tostring()

            # Create a dict with the data saved in the record files

            data = \
                {
                    'cie': wrap_bytes(cie_bytes),
                    'label': wrap_bytes(label)
                }

            # Wrap the data as TensorFlow Features.
            feature = tf.train.Features(feature=data)

            # Wrap again as a TensorFlow Example.
            example = tf.train.Example(features=feature)

            # Serialize the data.
            serialized = example.SerializeToString()

            # Write the serialized data to the TFRecords file.
            writer.write(serialized)



################################################################################
# Create compress objects for inputs and outputs
#


def save_input_output(path, newpath):
    image = image_util.read_image(path)
    lab = image_util.convert_rgb_to_lab(image)
    se = image_util.soft_encode_lab_img(lab)

    new_path = newpath
    np.savez_compressed(new_path, input=lab, output=se)


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
# Create dataset for the 2 class experiments
# This experiment will consist in training 2 classes, each with the custom weights,
# and with the other's custom weights
#
def save_input_output_2classes(path, newpath):
    image = image_util.read_image(path)
    lab = image_util.convert_rgb_to_lab(image)
    se = image_util.soft_encode_lab_img(lab)

    new_path = newpath
    np.savez_compressed(new_path, input=lab, output=se)

def create_twoclasses_id():
    classes = ['n01443537', 'n02099712']
    root_dir = '../data/tiny-imagenet-200/train/'

    train_ids = {}
    validation_ids = {}

    for cur_class in classes:
        dir_path = root_dir + cur_class + '/images/'
        train_ids[cur_class] = []
        for subdir, dir, files in os.walk(dir_path):
            for file in files:
                path = os.path.join(dir_path, file).replace('\\', '/')
                if path.split('.')[-1] == "JPEG" and not is_grey_image(path):
                    train_ids[cur_class].append(path)

    print('Number of training ids for fish', len(train_ids[classes[0]]))
    with open('./saved_objects/train_ids_fish_uncompressed.pickle', 'wb') as fp:
        pickle.dump(train_ids[classes[0]], fp)
    print('Number of training ids for dog', len(train_ids[classes[1]]))
    with open('./saved_objects/train_ids_dog_uncompressed.pickle', 'wb') as fp:
        pickle.dump(train_ids[classes[1]], fp)


    val_keys = load_validation_keys()
    root_dir = "../data/tiny-imagenet-200/val/images/"

    validation_ids[classes[0]] = []
    validation_ids[classes[1]] = []
    for image in val_keys:
        if val_keys[image] in classes[0]:
            path = os.path.join(root_dir, image).replace('\\', '/')
            validation_ids[classes[0]].append(path)
        elif val_keys[image] in classes[1]:
            path = os.path.join(root_dir, image).replace('\\', '/')
            validation_ids[classes[1]].append(path)

    print('Number of validation ids for fish', len(validation_ids[classes[0]]))
    with open('./saved_objects/validation_ids_fish_uncompressed.pickle', 'wb') as fp:
        pickle.dump(validation_ids[classes[0]], fp)
    print('Number of validation ids for dog', len(validation_ids[classes[1]]))
    with open('./saved_objects/validation_ids_dog_uncompressed.pickle', 'wb') as fp:
        pickle.dump(validation_ids[classes[1]], fp)

def save_twoclasses_npz():
    train_dir = './saved_objects/train_ids_'
    validation_dir = './saved_objects/validation_ids_'
    output_dir = '../data/2classes/'
    classes = ['fish', 'dog']

    training_ids_npz = {}
    validation_ids_npz = {}

    train_output_dir = output_dir + 'train/'
    for i in range(len(classes)):
        path_in = train_dir + classes[i] + '_uncompressed.pickle'
        training_ids_npz[classes[i]] = []
        with open(path_in, 'rb') as fp:
            ids = pickle.load(fp)

        for image in ids:
            newname = image.split('/')[-1].split('.')[0]
            newpath = train_output_dir + newname + '.npz'
            training_ids_npz[classes[i]].append(newpath)
            save_input_output_2classes(image, newpath)

        direc = train_dir + classes[i] + '.pickle'
        print(direc, 'has length', len(training_ids_npz[classes[0]]))
        with open(direc, 'wb') as fp:
            pickle.dump(training_ids_npz[classes[i]], fp)


    validation_output_dir = output_dir + 'val/'
    for i in range(len(classes)):
        path_in = validation_dir + classes[i] + '_uncompressed.pickle'
        validation_ids_npz[classes[i]] = []
        with open(path_in, 'rb') as fp:
            ids = pickle.load(fp)

        for image in ids:
            newname = image.split('/')[-1].split('.')[0]
            newpath = validation_output_dir + newname + '.npz'
            validation_ids_npz[classes[i]].append(newpath)
            save_input_output_2classes(image, newpath)

        direc = validation_dir + classes[i] + '.pickle'
        print(direc, 'has length', len(validation_ids_npz[classes[0]]))
        with open(direc, 'wb') as fp:
            pickle.dump(validation_ids_npz[classes[i]], fp)


def create_twoclasses_dataset():
    # create_twoclasses_id()
    save_twoclasses_npz()

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
    create_twoclasses_dataset()
