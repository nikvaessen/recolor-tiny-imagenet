################################################################################
# Utility methods related to image encoding, binning and distributions
# required for the implementation of the VGG 16
#
# author(s): Nik Vaessen, Merlin Sewina
################################################################################

import os
import time

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from keras import backend as K

from skimage import io, color, transform

################################################################################
# Basic functions to load / edit pictures
#

def read_image(fn: str):
    if os.name == 'nt':
        fn = fn.replace('\\', '/') # activate for Windows

    return io.imread(fn)

def save_image(fn, img):
    io.imsave(fn, img)

def plot_image(x: np.ndarray):
    plt.figure(0)
    plt.imshow(x)

    plt.show()

def convert_rgb_to_lab(img):
    return color.rgb2lab(img)

def convert_lab_to_rgb(img):
    return color.lab2rgb(img)

def rgb_to_gray(img):
    cie = convert_rgb_to_lab(img)
    gray = np.zeros(cie.shape)
    gray[:, :, 0] = cie[:, :, 0]
    gray = convert_lab_to_rgb(gray)
    return gray


























