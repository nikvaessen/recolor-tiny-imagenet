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

################################################################################
# weights used in multinomial loss_function

_weight_filename = 'waitlist.pickle'
_weight_path = os.path.join(_root_dir, _weight_filename)
weights = np.array(load_pickled_data(_weight_path))

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
tiny_imagenet_dataset_tiny = 'tiny-imagenet-dataset-tiny'

_training_set_tiny_file_name = 'train_ids_tiny.pickle'
_training_set_tiny_path = os.path.join(_root_dir, _training_set_tiny_file_name)
training_set_tiny_file_paths = load_pickled_data(_training_set_tiny_path)

_validation_set_tiny_file_name = 'validation_ids_tiny.pickle'
_validation_set_tiny_path = os.path.join(_root_dir, _training_set_tiny_file_name)
validation_set_tiny_file_paths = load_pickled_data(_training_set_tiny_path)

_test_set_tiny_file_name = 'test_ids_tiny.pickle'
_test_set_tiny_path = os.path.join(_root_dir, _training_set_tiny_file_name)
test_set_tiny_file_paths = load_pickled_data(_training_set_tiny_path)


################################################################################
# define valid loss functions

l2_loss = "l2_loss"

multinomial_loss = "multinomial_loss"
weighted_multinomial_loss = "weighted_multinomial_loss"
