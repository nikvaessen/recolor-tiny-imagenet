################################################################################
#
# Define project-wide constants to be defined at start-up
#
# Author(s): Nik Vaessen
################################################################################

import os

import numpy as np

from keras import backend as K

################################################################################
# Constants related to binning of the lab space


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


# Stores the center (a,b) value of each bin
_path_bincenters = "../np/bincenters.npz"
lab_bin_centers = _load_np_from_file(_path_bincenters,
                                     'lab_bin_centers')

tf_lab_bin_centers = K.constant(lab_bin_centers)

# Stores the bounding boxes (a_min, a_max, b_min, b_max) of each bin
_path_bins = "../np/bins.npz"
lab_bin_bounding_boxes = _load_np_from_file(_path_bins,
                                            'lab_bin_bounding_boxes')


# Store the number of bins
num_bins = len(lab_bin_centers)
assert len(lab_bin_centers) == len(lab_bin_bounding_boxes)


