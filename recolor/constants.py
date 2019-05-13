################################################################################
#
# Define project-wide constants to be defined at start-up
#
# Author(s): Nik Vaessen
################################################################################

import os
import pickle

import numpy as np

################################################################################
# helper function


def find_saved_objects():
    option1 = "./saved_objects"
    option2 = "../saved_objects"
    option3 = "../recolor/saved_objects"
    option4 = "recolor/saved_objects"

    for option in [option1, option2, option3, option4]:
        if os.path.isdir(option):
            return option

    raise ValueError("unable to find 'saved_objects' folder")


_root_dir = find_saved_objects()


def _load_np_from_file(path, name, extract_array=True):
    if os.path.exists(path):
        ob = np.load(path)
        if extract_array:
            return ob['arr_0']
        else:
            return ob
    else:
        raise ValueError(path, " was not found, constant",
                         name, "cannot be loaded")


def load_pickled_data(path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)


################################################################################
# Constants related to binning of the lab space


# Stores the center (a,b) value of each bin
_filename_bincenters = 'bincenters.npz'
_path_bincenters = os.path.join(_root_dir, _filename_bincenters)
lab_bin_centers = _load_np_from_file(_path_bincenters,
                                     'lab_bin_centers')

# Stores the bounding boxes (a_min, a_max, b_min, b_max) of each bin
_filename_bins = "bins.npz"
_path_bins = os.path.join(_root_dir, _filename_bincenters)
lab_bin_bounding_boxes = _load_np_from_file(_path_bins,
                                            'lab_bin_bounding_boxes')


# Store the number of bins
num_lab_bins = len(lab_bin_centers)
assert len(lab_bin_centers) == len(lab_bin_bounding_boxes)

#######################################_full_dataset_weight_filename = 'waitlist.pickle'
#########################################
# weights used in multinomial loss_function


_full_dataset_weight_filename = 'waitlist.pickle'
_full_dataset_weight_path = os.path.join(_root_dir, _full_dataset_weight_filename)
full_dataset_weights = np.array(load_pickled_data(_full_dataset_weight_path))
_weight_filename = 'waitlist_tiny.pickle'
_weight_path = os.path.join(_root_dir, _weight_filename)
weights = np.array(load_pickled_data(_weight_path))

################################################################################
# soft-encoding paths

soft_encoding_training_and_val_dir = "../data/soft_encoded_training_val_dir/"
soft_encoding_test_dir = "../data/soft_encoded_training_val_dir"
soft_encoding_filename_postfix = "_soft_encoded.npz"


################################################################################
# references to data (each is an array of relative file paths


# full tiny-imagenet-200
tiny_imagenet_dataset_full = 'tiny-imagenet-dataset-full'

_training_set_full_file_name = 'train_ids.pickle'
_training_set_full_path = os.path.join(_root_dir, _training_set_full_file_name)
training_set_full_file_paths = load_pickled_data(_training_set_full_path)

_validation_set_full_file_name = 'validation_ids.pickle'
_validation_set_full_path = os.path.join(_root_dir, _training_set_full_file_name)
validation_set_full_file_paths = load_pickled_data(_training_set_full_path)

_test_set_full_file_name = 'test_ids.pickle'
_test_set_full_path = os.path.join(_root_dir, _training_set_full_file_name)
test_set_full_file_paths = load_pickled_data(_training_set_full_path)

# subset of tiny-imagenet-200
tiny_imagenet_dataset_tiny = 'tiny-imagenet-dataset-tiny-uncompressed'

_training_set_tiny_uncompressed_file_name = 'train_ids_tiny.pickle'
_training_set_tiny_uncompressed_path = os.path.join(_root_dir, _training_set_tiny_uncompressed_file_name)
training_set_tiny_uncompressed_file_paths = load_pickled_data(_training_set_tiny_uncompressed_path)
n_training_set_tiny_uncompressed = 14892

_validation_set_tiny_uncompressed_file_name = 'validation_ids_tiny.pickle'
_validation_set_tiny_uncompressed_path = os.path.join(_root_dir, _training_set_tiny_uncompressed_file_name)
validation_set_tiny_uncompressed_file_paths = load_pickled_data(_training_set_tiny_uncompressed_path)

_test_set_tiny_uncompressed_file_name = 'test_ids_tiny_uncompressed.pickle'
_test_set_tiny_uncompressed_path = os.path.join(_root_dir, _training_set_tiny_uncompressed_file_name)
test_set_tiny_uncompressed_file_paths = load_pickled_data(_training_set_tiny_uncompressed_path)

# subset of tiny-imagenet-200
tiny_imagenet_dataset_tiny = 'tiny-imagenet-dataset-tiny'

_training_set_tiny_file_name = 'train_ids_npz.pickle'
_training_set_tiny_path = os.path.join(_root_dir, _training_set_tiny_file_name)
training_set_tiny_file_paths = load_pickled_data(_training_set_tiny_path)
n_training_set_tiny = 14892

_validation_set_tiny_file_name = 'validation_ids_npz.pickle'
_validation_set_tiny_path = os.path.join(_root_dir, _training_set_tiny_file_name)
validation_set_tiny_file_paths = load_pickled_data(_training_set_tiny_path)

_test_set_tiny_file_name = 'test_ids_npz.pickle'
_test_set_tiny_path = os.path.join(_root_dir, _training_set_tiny_file_name)
test_set_tiny_file_paths = load_pickled_data(_training_set_tiny_path)

# subset of tiny-imagenet-200
debug_dataset = 'debug-dataset'
_debug_num = 1
_debug_bs = 1
_debug_end = _debug_num*_debug_bs

training_set_debug_file_paths = training_set_tiny_file_paths[0:_debug_end]
print(training_set_debug_file_paths)

validation_set_debug_file_paths = training_set_debug_file_paths

test_set_tiny_debug_paths = test_set_tiny_file_paths[0:_debug_end]


################################################################################
# define valid loss functions

l2_loss = "l2_loss"

multinomial_loss = "multinomial_loss"
weighted_multinomial_loss = "weighted_multinomial_loss"
