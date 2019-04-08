################################################################################
# Utility methods related to image encoding, binning and distributions
# required for the implementation of the loss function of the CNN.
#
# author(s): Nik Vaessen, Merlin Sewina
################################################################################

import numpy as np
import matplotlib.pyplot as plt

from skimage import io, color

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

################################################################################
# converting, binning and encoding of bins as classes


def convert_rgb_to_lab(img):
    return color.rgb2lab(img)


def convert_lab_to_rgb(img):
    return color.lab2rgb(img)


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


def _test_image():
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
    img = _test_image()
    lab = convert_rgb_to_lab(img)

    print("rgb\n", img)
    print("lab\n", lab)

    r = one_hot_encode_lab_img(lab)
    print(r)



def main():
    # test_lab_conversion_scikit()
    # test_rgb2lab2binned2rgb()
    test_encoding()
    pass


if __name__ == '__main__':
    main()