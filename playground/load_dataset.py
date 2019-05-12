from recolor.constants import training_set_full_file_paths
from recolor.keras_util import load_image_grey_in_softencode_out, load_image_grey_in_ab_out

from recolor.constants import num_lab_bins

import numpy as np

import time


def __data_generation(batch_paths):
    # Initialization
    batch_size = len(batch_paths)
    dim_in = (64, 64, 3)
    dim_out = (64, 64, 2)

    X = np.empty((batch_size, *dim_in))
    y = np.empty((batch_size, *dim_out))

    # Generate data
    for i, path in enumerate(batch_paths):
        # Store sample
        print(i)
        inp, outp = load_image_grey_in_ab_out(path)
        X[i, ] = inp
        y[i, ] = outp

    return X, y


X, y = __data_generation(training_set_full_file_paths)

while True:
    time.sleep(10)
