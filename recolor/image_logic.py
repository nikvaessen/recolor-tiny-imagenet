################################################################################
# Utility methods related to image encoding, binning and distributions
# required for the implementation of the loss function of the CNN.
#
# author(s): Nik Vaessen, Merlin Sewina
################################################################################

import numpy as np

from keras.preprocessing import image

################################################################################

rgb_min = 0
rgb_max = 255

preferred_bin_size = 20


################################################################################
# image loading and display

test_image = "../test_images/test_image2.png"


def read_image(fn: str):
    return image.img_to_array(image.load_img(fn)).astype(np.int16)


def plot_image(x: np.ndarray):
    import matplotlib.pyplot as plt

    plt.figure(0)
    plt.imshow(x)

    plt.figure(1)
    y = bin_rgb(x)
    plt.imshow(y)

    eq = (x == y).all()
    print(eq)

    plt.show()


def bin_rgb(x, binsize=preferred_bin_size):
    return np.minimum(rgb_max, (((x // binsize) * binsize)
                                + (binsize // 2)).astype(np.int16))


def one_hot_encode_rgb_img(img: np.ndarray,
                           binsize=preferred_bin_size) -> np.ndarray:
    # potential improvement: do a batch of images at the same time
    bin = (rgb_max // binsize)

    r = img[:, :, 0] // binsize
    g = img[:, :, 1] // binsize
    b = img[:, :, 2] // binsize

    return r * bin**2 + g * bin + b


def soft_encode_rgb_img(img, n=5, binsize=preferred_bin_size):
    r = range(0, 255 // binsize)

    for ridx in r:
        for bidx in r:
            for gidx in r:
                pass


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


def main():
    # img = _test_image()
    # img = bin_rgb(img)
    # print(one_hot_encode_rgb_img(img))

    img = read_image(test_image)
    print(img.shape, np.max(img), np.min(img))
    plot_image(img)


if __name__ == '__main__':
    main()