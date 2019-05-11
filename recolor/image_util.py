################################################################################
# Utility methods related to image encoding, binning and distributions
# required for the implementation of the loss function of the CNN.
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

from . import constants as c

################################################################################
# constants related to the binning

rgb_min = 0
rgb_max = 255

lab_min = -128
lab_max = 127

rgb_preferred_bin_size = 20
lab_preferred_bin_size = 10

lab_bin_bounding_boxes = c.lab_bin_bounding_boxes
lab_bin_centers = c.lab_bin_centers
num_lab_bins = c.num_lab_bins

################################################################################
# image loading and display


def read_image(fn: str):
    fn = fn.replace('\\', '/') # activate for Windows
    return io.imread(fn)


def save_image(fn, img):
    io.imsave(fn, img)


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

    return img


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

def rgb_to_softencode(path):
    print(path)
    image = read_image(path)


def bin_lab_slow(img, bins=lab_bin_bounding_boxes):
    c = img[:, :, 1:]

    c_binned = np.copy(c)
    
    for i in range(c.shape[0]):
        #print("\r{} out of {}".format(i, c.shape[0]))
        for j in range(c.shape[1]):
            a, b = c[i, j, :].flat

            found = False
            for a_min, a_max, b_min, b_max in bins:
                if a_min <= a <= a_max and b_min <= b <= b_max:
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


def soft_encode_lab_img(img: np.ndarray,
                        bincenters=lab_bin_centers,
                        apply_kernel=False,
                        gaussian_kernel_var=5):
    """Given a lab image returns a soft encoding per pixel"""
    """current version takes about 7 sec per 64*64 image """
    n_rows = img.shape[0]
    n_cols = img.shape[1]
    n_pixels = n_rows * n_cols
    n_bins = len(bincenters)

    # using np operations ->
    # get a,b values as array
    a = (img[:, :, 1]).flatten()
    b = (img[:, :, 2]).flatten()

    # creates an array of length 'n_pixels', with a tuple value
    # (a, b) for each pixel
    ab = np.stack((a, b), axis=-1)

    # for each bin, compute the distance of each pixel in the image to the bin
    # stored in variable d:
    # row: each bin
    # column: each pixel (flattened 1d array of length n_pixels)
    d = np.zeros((n_bins, n_pixels))
    for idx, c in enumerate(bincenters):
        dist = np.linalg.norm(ab - c, axis=-1)
        d[idx, :] = dist

    # for each column, the (the array of pixels), we want to know the
    # index of the 5 smallest distances
    num = 5
    ind = np.argsort(d, axis=0)[:num, :]
    val = np.take(d, ind)

    # convert the closest 5 pixels into a probability distribution
    # results will contain 'n_pixel' numpy arrays of size 'n_bins'
    results = []
    for pixel_column_idx in range(0, n_pixels):
        bins_prob_dist = np.zeros(n_bins)

        indexes = []
        distances = []
        for i in range(0, num):
            correct_index = ind[i, pixel_column_idx]
            distance = val[i, pixel_column_idx]

            indexes.append(correct_index)
            distances.append(distance)

        # create a gaussian kernel to smooth distances
        if apply_kernel:
            mean = np.mean(distances)
            var = gaussian_kernel_var
            pdf = stats.norm(mean, var).pdf
            distances = pdf(distances)

        # store distances in 5-hot vector
        for i, dist_idx in enumerate(indexes):
            bins_prob_dist[dist_idx] = distances[i]

        # normalize 5-hot vector to make it a probability distribution
        bins_prob_dist /= bins_prob_dist.sum()

        results.append(bins_prob_dist)

    # transform the results back into an image shape
    r = np.zeros((n_pixels, n_bins))
    for idx, color_prob_dist in enumerate(results):
        if np.isclose(1, color_prob_dist.sum()):
            r[idx, :] = color_prob_dist
        else:
            raise ValueError("calculated a prob dist which did not sum to 1")

    r = r.reshape((n_rows, n_cols, n_bins))

    return r


def probability_dist_to_ab(pdist, T=1):
    # pdist is assumed to have shape:
    # (batch_size, image_height, image_width, n_bins)

    assert len(pdist.shape) == 4
    print(pdist.shape)
    batch_ab = np.empty((*pdist.shape[0:3], 2))

    for i in range(pdist.shape[0]):
        p = pdist[i, :, :, :]

        # p -= np.max(p, axis=2)[:, :, np.newaxis]
        # p = np.exp(np.log(p)/T)
        # p /= p.sum()

        bin_indexes = np.argmax(p, axis=2)
        print(bin_indexes.shape)
        for j in range(p.shape[0]):
            for k in range(p.shape[1]):
                bin_idx = bin_indexes[j, k]
                ab = lab_bin_centers[bin_idx]
                batch_ab[i, j, k, :] = ab

    return batch_ab


# tensorflow implementation of `probability_dist_to_ab`
def probability_dist_to_ab_tensor(pdist):
    # pdist is assumed to have shape:
    # (batch_size, image_height, image_width, n_bins)

    assert len(pdist.shape) == 4

    batch_ab = K.zeros((pdist.shape[0], pdist.shape[1], pdist.shape[2], 2))
    bin_indexes = K.argmax(pdist, axis=3)
    print('bin indexes', bin_indexes)
    bin_coords = K.constant(lab_bin_centers)
    K.map_fn()

    for i in range(pdist.shape[0]):
        p = pdist[i, :, :, :]

        # p -= np.max(p, axis=2)[:, :, np.newaxis]
        # p = np.exp(np.log(p)/T)
        # p /= p.sum()

        bin_indexes = K.argmax(p, axis=2)
        print(bin_indexes.shape)
        for j in range(p.shape[0]):
            for k in range(p.shape[1]):
                bin_idx = bin_indexes[j, k]
                ab = bin_coords[bin_idx]
                batch_ab[i, j, k, :] = ab

    return batch_ab


###############################################################################
# tests

test_image = "../test_images/test_image2.png"


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

    for idx, (a_min, a_max, b_min, b_max) in enumerate(lab_bin_bounding_boxes):
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


def create_bin_center_file():
    """Calculate the bin centers and save them"""
    result = []
    for b in lab_bin_bounding_boxes:
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
    print(type(lab_bin_bounding_boxes))
    for b in lab_bin_bounding_boxes:
        print(b)
    print("length: ", len(lab_bin_bounding_boxes))

    img = read_image(test_image)
    print(".jpeg read", img.shape, img.dtype, np.max(img), np.min(img))

    lab = convert_rgb_to_lab(img)
    print("lab", lab.shape, lab.dtype, np.max(lab), np.min(lab))
    binned = bin_lab(lab)

    l_equal = lab[:, :, 0] == binned[:, :, 0]
    print(l_equal.all())

    rgb = convert_lab_to_rgb(binned)

    plot_img_converted(img, lambda x: rgb)


def test_lab_5encode():
    img = read_image(test_image)

    img = transform.resize(img, (64, 64))
    lab = convert_rgb_to_lab(img)

    start = time.time()
    r = soft_encode_lab_img(lab)
    end = time.time()

    print("time", end-start, "seconds")

    result = r[0]
    colors = []
    binned_img = bin_lab_slow(lab)
    binned_img = convert_lab_to_rgb(binned_img)


    for idx in range(0, result.shape[0]):
        v = result[idx]
        if v > 0:
            c = lab_bin_centers[idx]
            colors.append(c)
    nbin = []
    for v in r:
        i = np.where( v==np.min(v[np.nonzero(v)]))
        nbin.append(lab_bin_centers[i[0][0]])
    print(colors)

    bimg = np.ones((64,64,3))*50
    for x in range(64):
        for y in range(64):
            bimg[x,y,1:] = nbin[(x*63)+y]

    image = np.ones((5, 4, 3)) * 50
    image[0, 0, 1:] = colors[0]
    image[1, 0, 1:] = colors[1]
    image[2, 0, 1:] = colors[2]
    image[3, 0, 1:] = colors[3]
    image[4, 0, 1:] = colors[4]

    image[0, 1, 1:] = lab[0, 0, 1:]
    image[1, 1, 1:] = lab[0, 0, 1:]
    image[2, 1, 1:] = lab[0, 0, 1:]
    image[3, 1, 1:] = lab[0, 0, 1:]
    image[4, 1, 1:] = lab[0, 0, 1:]

    image[0, 2, 1:] = binned_img[0, 0, 1:]
    image[1, 2, 1:] = binned_img[0, 0, 1:]
    image[2, 2, 1:] = binned_img[0, 0, 1:]
    image[3, 2, 1:] = binned_img[0, 0, 1:]
    image[4, 2, 1:] = binned_img[0, 0, 1:]
    
    image[0, 3, 1:] = lab_bin_centers[78]
    image[1, 3, 1:] = lab_bin_centers[95]
    image[2, 3, 1:] = lab_bin_centers[96]
    image[3, 3, 1:] = lab_bin_centers[111]
    image[4, 3, 1:] = lab_bin_centers[112]

    rgb = convert_lab_to_rgb(image)
    rgb2 = convert_lab_to_rgb(bimg)

    #plot_image(img)
    #plot_image(binned_img)
    # plot_image(lab)
    plot_image(rgb)
    plot_image(img)
    plot_image(rgb2)


def test_lab_encode():
    img = read_image(test_image)
    img = transform.resize(img, (64, 64))
    img = color.rgba2rgb(img)

    img = convert_rgb_to_lab(img)

    # plot_image(img)
    start = time.time()

    encoded = soft_encode_lab_img(img)

    for row in range(encoded.shape[0]):
        for col in range(encoded.shape[1]):
            dist = encoded[row, col, :]
            if not np.isclose(1, dist.sum()):
                print(dist)
                print(dist.sum())
                exit()

    end = time.time()
    total = end - start

    print("encoding took", total, "seconds")


def test_pdist_to_ab():
    img = read_image(test_image)
    img = transform.resize(img, (64, 64))
    img = color.rgba2rgb(img)

    lab = convert_rgb_to_lab(img)
    l = lab[:, :, 0]

    encoded = soft_encode_lab_img(lab)
    encoded = encoded.reshape(1, *encoded.shape)

    ab_predicted = probability_dist_to_ab(encoded)

    new_lab = np.empty((64, 64, 3))
    new_lab[:, :, 0] = l
    new_lab[:, :, 1:] = ab_predicted

    new_rgb = convert_lab_to_rgb(new_lab)
    plot_image(new_rgb)


def plot_all_bins():
    lab = np.ones((1, 289, 3)) * 0
    # lab = lab.flatten()

    for idx, b in enumerate(lab_bin_centers):
        # start = idx * 3
        # end = (idx+1) * 3
        # print(b)
        # lab[start:end] = (50, b[0], b[1])
        lab[0, idx, 0:] = (50, *b)

    lab = lab.reshape(17, 17, 3)

    rgb = convert_lab_to_rgb(lab)

    plot_image(rgb)


def main():
    # test_lab_conversion_scikit()
    # test_rgb2lab2binned2rgb()
    # test_encoding()
    # test_lab_bounds()
    # test_lab_bounds_inverted()
    # create_bin_numpy_file()
    # create_bin_center_file()
    # test_bins()
    # test_lab_5encode()
    # test_lab_encode()
    # test_pdist_to_ab()
    plot_all_bins()
    pass


if __name__ == '__main__':
    # main()
    path = '../data/tiny-imagenet-200/train/n02058221/images/n02058221_3.JPEG'
    rgb_to_softencode(path)
