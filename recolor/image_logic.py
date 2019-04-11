################################################################################
# Utility methods related to image encoding, binning and distributions
# required for the implementation of the loss function of the CNN.
#
# author(s): Nik Vaessen, Merlin Sewina
################################################################################

import math
import os

import numpy as np
import matplotlib.pyplot as plt

from skimage import io, color
from color import colorconv as color_custom

################################################################################

rgb_min = 0
rgb_max = 255

lab_min = -128
lab_max = 127

rgb_preferred_bin_size = 20
lab_preferred_bin_size = 10

################################################################################
# image loading and display

test_image = "../test_images/test_image2.png"


def read_image(fn: str):
    return io.imread(fn)[:, :, 0:3]


def plot_image(x: np.ndarray):
    plt.figure(0)
    plt.imshow(x)

    plt.show()


def plot_img_converted(img, converter):
    plt.figure(0)
    plt.imshow(img)

    plt.figure(1)
    plt.imshow(converter(img))

    plt.show()


def plot_ingamut(bins, bin_ingamut):
    bin_size = lab_preferred_bin_size
    ab_range = range(lab_min, lab_max, bin_size)

    img_l = np.zeros((len(ab_range), len(ab_range)))
    img_a = np.copy(img_l)
    img_b = np.copy(img_l)

    for idx, bounds in enumerate(bins):
        if bin_ingamut[idx] == 0:
            img_l.flat[idx] = 100
            img_a.flat[idx] = 0
            img_b.flat[idx] = 0
            continue

        a_min, a_max, b_min, b_max = bounds

        a = a_min + (1 / 2 * bin_size)
        b = b_min + (1 / 2 * bin_size)

        img_l.flat[idx] = 50
        img_a.flat[idx] = a
        img_b.flat[idx] = b

    img = np.ones((img_a.shape[0], img_a.shape[1], 3))
    img[:, :, 0] = img_l
    img[:, :, 1] = img_a
    img[:, :, 2] = img_b

    rgb_ingamut = convert_lab_to_rgb(img)

    print("bins ingamut: ", bin_ingamut.sum())
    plot_image(rgb_ingamut)


################################################################################
# converting, binning and encoding of bins as classes


def convert_rgb_to_lab(img):
    return color.rgb2lab(img)


def convert_lab_to_rgb(img):
    return color.lab2rgb(img)


def custom_lab2rgb(lab):
    return color_custom.lab2rgb(lab, clip=False)


def bin_rgb(x, binsize=rgb_preferred_bin_size):
    return np.minimum(rgb_max, (((x // binsize) * binsize)
                                + (binsize // 2)).astype(np.int16))


def one_hot_encode_rgb_img(img: np.ndarray,
                           binsize=rgb_preferred_bin_size) -> np.ndarray:
    # potential improvement: do a batch of images at the same time
    bin = (rgb_max // binsize)

    r = img[:, :, 0] // binsize
    g = img[:, :, 1] // binsize
    b = img[:, :, 2] // binsize

    return r * bin**2 + g * bin + b


def soft_encode_rgb_img(img, n=5, binsize=rgb_preferred_bin_size):
    raise NotImplementedError()


def bin_lab(img, binsize=lab_preferred_bin_size):
    c = img[:, :, 1:]

    binned_color = ((c // binsize) * binsize) + (binsize // 2)
    binned_color = np.minimum(lab_max, binned_color)
    binned_color = np.maximum(lab_min, binned_color)

    img[:, :, 1:] = binned_color

    return img


def one_hot_encode_lab_img(img: np.ndarray,
                           binsize=lab_preferred_bin_size):
    bin = abs(lab_max - lab_min) // binsize

    a = (img[:, :, 1] + 128)
    print(a)
    a = a // binsize
    print(a)

    b = (img[:, :, 2] + 128) // binsize

    return a * bin + b


###############################################################################
# tests


def _get_test_image():
    img = np.ones((1, 5, 3))
    img[0, 0, 0] = 255
    img[0, 0, 1] = 0
    img[0, 0, 2] = 0

    img[0, 1, 1] = 255
    img[0, 1, 0] = 0
    img[0, 1, 2] = 0

    img[0, 2, 2] = 255
    img[0, 2, 1] = 0
    img[0, 2, 0] = 0

    img[0, 3, 2] = 255
    img[0, 3, 1] = 255
    img[0, 3, 0] = 255

    img[0, 4, 2] = 0
    img[0, 4, 1] = 0
    img[0, 4, 0] = 0

    return img


def test_lab_bounds():
    bin_size = lab_preferred_bin_size
    ab_range = range(lab_min, lab_max, bin_size)
    l_range = range(0, 101)

    bins = []

    for i in ab_range:
        for j in ab_range:
            b = (i, i + bin_size - 1, j, j + bin_size - 1)
            bins.append(b)

    bin_ingamut = np.zeros((len(bins)))

    # print(len(bins))
    # print(len(ab_range))

    def in_gamut(r, g, b):
        r_edge = r < 0 or r > 1
        g_edge = g < 0 or g > 1
        b_edge = b < 0 or b > 1

        return not (r_edge or g_edge or b_edge)

    for idx, bounds in enumerate(bins):
        print("bin {:3d} out of {}".format(idx + 1, len(bins)))
        a_min, a_max, b_min, b_max = bounds

        img_a = np.zeros((bin_size, bin_size))
        img_b = np.zeros(img_a.shape)

        img_a[:, :] = np.array((range(a_min, a_max + 1)))
        img_b[:, :] = np.array((range(b_min, b_max + 1)))
        img_b = img_b.T

        for l in l_range:
            if bin_ingamut.flat[idx] == 1:
                break

            img_l = np.ones(img_a.shape) * l

            img = np.zeros((img_l.shape[0], img_l.shape[1], 3))
            img[:, :, 0] = img_l
            img[:, :, 1] = img_a
            img[:, :, 2] = img_b

            rgb = custom_lab2rgb(img)

            for i in range(0, rgb.shape[0]):
                if bin_ingamut.flat[idx] == 1:
                    break

                for j in range(0, rgb.shape[1]):
                    r, g, b = rgb[i, j, :]

                    if in_gamut(r, g, b):
                        bin_ingamut.flat[idx] = 1
                        break

        plot_ingamut(bin_ingamut)


def test_lab_bounds_inverted():
    total = 255**3

    rgb_all = np.zeros((1, total, 3))

    count = 0
    for r in range(rgb_min, rgb_max):
        print("\r{:3d} out of {}".format(r+1, rgb_max), end="", flush=True)
        for g in range(rgb_min, rgb_max):
            for b in range(rgb_min, rgb_max):
                rgb_all[:, count, :] = (r, g, b)
                count += 1
    print()

    lab = convert_rgb_to_lab(rgb_all)

    bin_size = lab_preferred_bin_size
    ab_range = range(lab_min, lab_max, bin_size)

    bins = []

    for i in ab_range:
        for j in ab_range:
            b = (i, i + bin_size - 1, j, j + bin_size - 1)
            bins.append(b)

    dir = "../np/"
    fn = "in_gamut.npy"
    path = os.path.join(dir, fn)
    if os.path.exists(path):
        bin_ingamut = np.load(fn)
    else:
        if not os.path.exists(dir):
            os.mkdir(dir)

        bin_ingamut = np.zeros((len(bins)))

        for i in range(0, total):
            if i % 100000 == 0:
                np.save(path, bin_ingamut)
                print("\r{:7d} out of {}".format(i, total),
                      end="", flush=True)

            l, a, b = lab[:, i, :].flat

            for idx, (a_min, a_max, b_min, b_max) in enumerate(bins):
                if a_min < a < a_max and b_min < b < b_max:
                    bin_ingamut.flat[idx] = 1
                    break

        print()
        np.save(path, bin_ingamut)

    plot_ingamut(bins, bin_ingamut)


def test_lab_conversion_scikit():
    img = read_image(test_image)
    lab = color.rgb2lab(img)
    rgb = color.lab2rgb(lab)

    print("img", img.shape, np.min(img), np.max(img), img.dtype)
    print("lab", lab.shape, np.min(lab), np.max(lab), lab.dtype)
    print("rgb", rgb.shape, np.min(rgb), np.max(rgb), rgb.dtype)


def test_rgb2lab2binned2rgb():
    img = read_image(test_image) / 255

    def convert(img):
        lab = convert_rgb_to_lab(img)
        binned = bin_lab(lab)
        rgb = convert_lab_to_rgb(binned)

        return rgb

    plot_img_converted(img, convert)


def test_encoding():
    img = _get_test_image()
    lab = convert_rgb_to_lab(img)

    print("rgb\n", img)
    print("lab\n", lab)

    r = one_hot_encode_lab_img(lab)
    print(r)


def main():
    # test_lab_conversion_scikit()
    # test_rgb2lab2binned2rgb()
    # test_encoding()
    # test_lab_bounds()
    test_lab_bounds_inverted()
    pass


if __name__ == '__main__':
    main()