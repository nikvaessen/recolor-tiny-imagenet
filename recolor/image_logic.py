################################################################################
# Utility methods related to image encoding, binning and distributions
# required for the implementation of the loss function of the CNN.
#
# author(s): Nik Vaessen, Merlin Sewina
################################################################################

import os

import numpy as np
import matplotlib.pyplot as plt

from skimage import io, color

################################################################################
# constants related to the binning

rgb_min = 0
rgb_max = 255

lab_min = -128
lab_max = 127

rgb_preferred_bin_size = 20
lab_preferred_bin_size = 10

path_rgb_bin_indexes = "../np/rgb_bin_indexes.npz"
path_bincenters = "../np/bincenters.npz" 
path_bins = "../np/bins.npz"

if os.path.exists(path_bins):
    bins = np.load(path_bins)['arr_0']
else:
    print("WARNING: ", path_bins, " was not found, some methods in ", __name__,
          "will fail")

if os.path.exists(path_bincenters):
    bincenters = np.load(path_bincenters)['arr_0']
else:
    print("WARNING: ", path_bincenters, " was not found, some methods in ", __name__,
          "will fail")

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
    img_a = np.zeros((len(ab_range), len(ab_range)))
    img_b = np.zeros((len(ab_range), len(ab_range)))

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

    print("bins ingamut: ", len(set(bin_ingamut)))
    plot_image(rgb_ingamut)
    return(img)


################################################################################
# converting, binning and encoding of bins as classes

def _get_potential_bins():
    bin_size = lab_preferred_bin_size
    ab_range = range(lab_min, lab_max, bin_size)

    bins = []

    for i in ab_range:
        for j in ab_range:
            b = (i, i + bin_size, j, j + bin_size)
            bins.append(b)

    return bins


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

# depricated
def soft_encode_rgb_img(img, n=5, binsize=rgb_preferred_bin_size):
    raise NotImplementedError()


def bin_lab_slow(img, bins=bins):
    c = img[:, :, 1:]

    c_binned = np.copy(c)

    for i in range(c.shape[0]):
        print("\r{} out of {}".format(i, c.shape[0]))
        for j in range(c.shape[1]):
            a, b = c[i, j, :].flat

            found = False
            for a_min, a_max, b_min, b_max in bins:
                if a_min < a < a_max and b_min < b < b_max:
                    a = a_min + 1/2 * lab_preferred_bin_size
                    b = b_min + 1/2 * lab_preferred_bin_size
                    c_binned[i, j, :] = (a, b)
                    found = True

            if not found:
                raise ValueError("unable to bin {} and {}".format(a, b))

    img[:, :, 1:] = c_binned

    return img


def bin_lab(img):
    c = np.copy(img[:, :, 1:])
    sign = c >= 0

    np.floor_divide(c, lab_preferred_bin_size, out=c)
    np.multiply(c, lab_preferred_bin_size, out=c)
    np.add(c, lab_preferred_bin_size/2, out=c, where=(c >= 0))
    np.subtract(c, lab_preferred_bin_size/2, out=c, where=~sign)

    img[:, :, 1:] = c

    return img


def one_hot_encode_lab_img(img: np.ndarray,
                           binsize=lab_preferred_bin_size):
    bin = abs(lab_max - lab_min) // binsize

    a = (img[:, :, 1] + abs(lab_min))
    #print(a)
    a = a // binsize
    #print(a)

    b = (img[:, :, 2] + abs(lab_min)) // binsize

    return a * bin + b

"""Given a lab image returns a soft encoding per pixel"""
def soft_encode_lab_img(img: np.ndarray, binsize=lab_preferred_bin_size, bincenters=bincenters):
    inGamutBinAmount = len(bincenters)

    # using np operations ->
    # get a,b values as array
    a = (img[:,:,1]).flatten()
    b = (img[:,:,2]).flatten()
    ab = np.stack((a,b),axis=-1)
    # calculate distance to all bin centers per pixel

    # per pixel
    # 5 times, select min distance, remeber distance and bin index, set this bins distance to high number
    # set index of remeberd bins to bindistance/sumbindistance
    # gaussian kernel thingy?!
    return ''


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
    from color.colorconv import lab2rgb as custom_lab2rgb

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

        plot_ingamut(bins, bin_ingamut)


def test_lab_bounds_inverted():
    rgb_tuples = []

    for r in range(rgb_min, rgb_max + 1):
        print("\r{:3d} out of {}".format(r + 1, rgb_max), end="", flush=True)
        for g in range(rgb_min, rgb_max + 1):
            for b in range(rgb_min, rgb_max + 1):
                rgb_tuples.append((r, g, b))
    print()

    total = len(rgb_tuples)
    rgb = np.zeros((1, total, 3))

    for idx, (r, g, b) in enumerate(rgb_tuples):
        rgb[:, idx, :] = (r/255, g/255, b/255)

    print("created rgb image. dtype=", rgb.dtype)
    for i in range(0, 3):
        maxi = np.max(rgb[:, :, i])
        mini = np.min(rgb[:, :, i])
        print("channel {}, max={}, min={}".format(i, maxi, mini))

    lab = convert_rgb_to_lab(rgb)

    lab_tuples = []
    for i in range(0, len(rgb_tuples)):
        l, a, b = lab[:, i, :].flat
        lab_tuples.append((l, a, b))

    print("created lab image. dtype=", lab.dtype)
    for i in range(0, 3):
        maxi = np.max(lab[:, :, i])
        mini = np.min(lab[:, :, i])
        print("channel {}, max={}, min={}".format(i, maxi, mini))

    print("converted to lab")

    # Multithread the gamut-checking
    import multiprocessing

    pool = multiprocessing.Pool(12)
    result = pool.map(func, lab_tuples)

    print(result)
    arr = np.array(result)
    path = path_rgb_bin_indexes
    np.savez_compressed(path, arr)


def func(lab_tuple):
    l, a, b = lab_tuple

    l_edge = l < 0 or l > 100
    a_edge = a < lab_min or a > lab_max
    b_edge = b < lab_min or b > lab_max

    if l_edge or a_edge or b_edge:
        return -1

    for idx, (a_min, a_max, b_min, b_max) in enumerate(bins):
        if a_min < a < a_max and b_min < b < b_max:
            return idx

    return -2, (l, a, b)


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


"""Calculate the bin centers and save them"""
def create_bin_center_file():
    result = []
    for b in bins:
        amin = b[0]
        amax = b[1]
        bmin = b[2]
        bmax = b[3]
        aavg = (amin+amax) // 2
        bavg = (bmin+bmax) // 2
        result.append([aavg,bavg])

    arr = np.array(result)
    path = path_bincenters
    np.savez_compressed(path, arr)

def create_bin_numpy_file():
    path = path_rgb_bin_indexes

    if not os.path.exists(path):
        print("place rgb_bin_indexes into ../np/ ( by running t"
              "est_lab_bounds_inverted() )")
        return

    r = np.load(path)['arr_0']

    print(r)
    print(r.shape)
    print(np.unique(r).shape)
    print(np.max(r), np.min(r))
    potential_bins = _get_potential_bins()

    indexes = np.unique(r).flat
    bins = []

    for idx in indexes:
        print(potential_bins[idx])
        bins.append(potential_bins[idx])

    np.savez_compressed(path_bins, bins)
    np.savez_compressed(path_rgb_bin_indexes, r)


def test_bins():
    print(type(bins))
    for b in bins:
        print(b)
    print("length: ", len(bins))

    img = read_image(test_image)
    print(".jpeg read", img.shape, img.dtype, np.max(img), np.min(img))

    lab = convert_rgb_to_lab(img)
    print("lab", lab.shape, lab.dtype, np.max(lab), np.min(lab))
    binned = bin_lab(lab)

    l_equal = lab[:, :, 0] == binned[:, :, 0]
    print(l_equal.all())

    rgb = convert_lab_to_rgb(binned)

    plot_img_converted(img, lambda x: rgb)


def main():
    # test_lab_conversion_scikit()
    # test_rgb2lab2binned2rgb()
    # test_encoding()
    # test_lab_bounds()
    # test_lab_bounds_inverted()
    # create_bin_numpy_file()
    # create_bin_center_file()
    # test_bins()
    soft_encode_lab_img(read_image(test_image))

    pass


if __name__ == '__main__':
    main()
