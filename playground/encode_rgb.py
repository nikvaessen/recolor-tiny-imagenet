from recolor import constants as c
from recolor import image_util

import os

import numpy as np

training_paths = c.training_set_tiny_file_paths
validation_paths = c.validation_set_tiny_file_paths


# for training_path in training_paths:
#     print(training_path)
#     path_split = os.path.split(training_path)
#     fn = path_split[1]
#
#     tag = fn.split("_")[0]
#     num = fn.split("_")[1]
#
#     real_path = os.path.join(
#         "..",
#         "data",
#         "tiny-imagenet-200",
#         "train",
#         tag,
#         "images",
#         "{}_{}.JPEG".format(tag, num)
#     )
#
#     rgb = image_util.read_image(real_path)
#
#     rgb_save_path = os.path.join(
#         path_split[0],
#         "{}_{}_rgb.npz".format(tag, num)
#     )
#
#     np.savez(rgb_save_path, rgb)
#
#     print("saving rgb to", rgb_save_path)


for validation_path in validation_paths:
    print(validation_path)
    path_split = os.path.split(validation_path)
    fn = path_split[1]

    tag = fn.split("_")[0]
    num = fn.split("_")[1]

    real_path = os.path.join(
        "..",
        "data",
        "tiny-imagenet-200",
        "val",
        "images",
        "{}_{}.JPEG".format(tag, num)
    )

    rgb = image_util.read_image(real_path)

    rgb_save_path = os.path.join(
        path_split[0],
        "{}_{}_rgb.npz".format(tag, num)
    )

    np.savez(rgb_save_path, rgb)

    print("saving rgb to", rgb_save_path)
