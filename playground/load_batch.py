from recolor.constants import training_set_tiny_file_paths
from recolor.keras_util import load_image_grey_in_softencode_out

import recolor.image_util as image_util

import numpy as np
import time

batch_size = 64

np.random.shuffle(training_set_tiny_file_paths)

filenames_npz = ['tuple_{}.npz'.format(idx) for idx in range(batch_size)]
filenames_comprez = ['tuple_{}_compr.npz'.format(idx) for idx in range(batch_size)]


def create_files():
    for idx, file in enumerate(training_set_tiny_file_paths[0:batch_size]):
        rgb = image_util.read_image(file)
        lab = image_util.convert_rgb_to_lab(rgb)

        soft_encode = image_util.soft_encode_lab_img(lab)

        np.savez('tuple_{}'.format(idx), lab, soft_encode)
        np.savez("tuple_{}_compr".format(idx), lab, soft_encode)


def load_test_disk():
    start = time.time()

    for file in training_set_tiny_file_paths[0:batch_size]:
        load_image_grey_in_softencode_out(file)

    end = time.time()

    elapsed = end - start

    print("took {} seconds to load a batch of size {} from disk".format(elapsed, batch_size))


def load_test_npz():
    start = time.time()

    for file in filenames_npz:
        arr = np.load(file)
        x, y = arr['arr_0'], arr['arr_1']

    end = time.time()

    elapsed = end - start

    print("took {} seconds to load a batch of size {} from npz files".format(elapsed, batch_size))


def load_test_npz_comprez():
    start = time.time()

    for file in filenames_comprez:
        arr = np.load(file)
        x, y = arr['arr_0'], arr['arr_1']
        # print([k for k in arr.keys()])

    end = time.time()

    elapsed = end - start

    print("took {} seconds to load a batch of size {} from comprezzed npz files".format(elapsed, batch_size))


create_files()
load_test_disk()
load_test_npz()
load_test_npz_comprez()
