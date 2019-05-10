################################################################################
# Utility methods related to Calculate the probabilities of all colours
# required to generate the data batches at training time
#
# author(s): Jade Cock
################################################################################
import pickle
import os
from image_logic import *
import copy
from matplotlib import pyplot as plt
from scipy.stats import norm

class ColourProbability():
    def __init__(self, num_label, label):
        '''
        :param num_label: Label reference given by the file words.txt in the dataset
        :param label: Text label
        '''
        self.num_label = num_label
        self.label = label
        # if colours is (R:25, G:45, B:65) -> absolutes[25, 45, 65]
        self.absolutes = np.zeros((256, 256, 256))  # Absolute counts of pixels
        self.probabilities = np.zeros((256, 256, 256))  # Probabilities of pixels
        self.counts = 0  # Number of pixels that have been accounted for so far

    def create_probabilities(self):
        '''
        :return: This function turns the absolute counts of pixel frequency into a probability
        '''
        self.probabilities /= self.counts

    def add_image(self, dim_x, dim_y, image):
        '''
        :param dim_x: first dimension of the image
        :param dim_y: second dimension of the image
        :param image: np.array [dim_x, dim_y, 3]
                First channel is the R value
                Second channel is the G value
                Third channel is the B value
        :return: Update the probability matrix and the absolute count matrix
        '''
        for i in range(dim_x):
            for j in range(dim_y):
                r = image[i, j, 0]
                g = image[i, j, 1]
                b = image[i, j, 2]
                self._add_rgb(r, g, b)

    def _add_rgb(self, r, g, b):
        '''
        :param r: R value of the pixel
        :param g: G value of the pixel
        :param b: B value of the pixel
        :return: Update the absolute counts and the probability values
        '''
        self.absolutes[r, g, b] += 1
        self.probabilities[r, g, b] += 1
        self.counts += 1


def test():
    '''
    This function tests that the above functions are functional
    '''
    colour_probs = ColourProbability()
    colour_probs.iterate('../Dataset/tiny-imagenet-200/words.txt', 'C:/Users/jadec/My Tresors/Stockholm/KTH/Block 4/Deep learning/Project/DLIDS/Dataset/Temporary')

    print("*" * 100)
    print('DEBUG')
    print(colour_probs.absolutes['n01443537'][255, 137, 192])
    print(colour_probs.absolutes['n01443537'][255, 124, 189])
    print(colour_probs.absolutes['n01443537'][255, 124, 201])
    print(np.sum(colour_probs.absolutes['n01443537']))
    print('_' * 50)
    print(colour_probs.absolutes['n01768244'][124, 144, 83])
    print(colour_probs.absolutes['n01768244'][123, 143, 80])
    print(colour_probs.absolutes['n01768244'][119, 141, 77])
    print(np.sum(colour_probs.absolutes['n01768244']))
    print('_' * 50)
    print(colour_probs.absolutes['n01882714'][62, 60, 65])
    print(colour_probs.absolutes['n01882714'][207, 205, 210])
    print(colour_probs.absolutes['n01882714'][147, 145, 150])
    print(np.sum(colour_probs.absolutes['n01882714']))


def load_keys():
    '''
    This function loads the file keeping track of the labels of the image
    key - > numerical value of the file (also the name of the folder they are in)
    value - > text value of the file
    '''
    label_path = "../Dataset/tiny-imagenet-200/words.txt"
    keys = {}
    with open(label_path) as f:
        for line in f:
            key, val = line.split('\t')
            keys[key] = val
    return keys


def compute_probabilities():
    '''
    Loop through all the images in the stanford tiny image dataset and updates the pixel counts
    '''
    all_probs = ColourProbability('all', 'all') # Contains loss regardless of the probability
    file_counter = 0
    labels = load_keys()
    train_path = "../Dataset/tiny-imagenet-200/train"
    counter_gray = 0

    for subdirs, dirs, files in os.walk(train_path):
        if len(files) == 500:
            file_counter += 1

            label = files[0][:9]
            label_name = labels[label]
            print(file_counter, ': ', label_name)
            colour_probs = ColourProbability(label, label_name)

            for file in files:
                path = os.path.join(subdirs, file)
                try:
                    image = read_image(path)
                    colour_probs.add_image(image.shape[0], image.shape[1], image)
                    all_probs.add_image(image.shape[0], image.shape[1], image)
                except IndexError:
                    counter_gray += 1

            colour_probs.create_probabilities()
            path = '../probabilities/probability_object_' + label + '.pickle'
            with open(path, 'wb') as fp:
                pickle.dump(colour_probs, fp)

    print(counter_gray)
    all_probs.create_probabilities()
    path = '../probabilities/probability_object_all.pickle'
    with open(path, 'wb') as fp:
        pickle.dump(all_probs, fp)


def create_rgb_to_bin():
    '''
    Create a dictionary of all rgb colours to the corresponding bin size
    '''

    path_bins = '../np/bins.npz'
    bins = np.load(path_bins)['arr_0']
    n_bins = len(bins)

    path_bincenters = '../np/bincenters.npz'
    bincenters = np.load(path_bincenters)['arr_0']

    rgb_to_bin = {}

    for rall in range(256):
        print(rall)
        for gall in range(256):
            for ball in range(256):
                r = rall / 256
                g = gall / 256
                b = ball / 256

                r = np.ones((1, 1)) * r
                g = np.ones((1, 1)) * g
                bb = np.ones((1, 1)) * b

                rgb_pixel = np.stack((r, g, bb), axis=-1)
                cie_pixel = convert_rgb_to_lab(rgb_pixel)

                a = cie_pixel[:, :, 1].flatten()
                b = cie_pixel[:, :, 2].flatten()

                ab = np.stack((a, b), axis=-1)
                d = np.zeros(n_bins)

                for idx, c in enumerate(bincenters):
                    dist = np.linalg.norm(ab - c, axis=-1)
                    d[idx] = dist

                bin = np.argmin(d)

                r *= 256
                g *= 256
                bb *= 256
                rgb_string = str(r[0][0]) + ', ' + str(g[0][0]) + ', ' + str(bb[0][0])
                print(rgb_string)
                rgb_to_bin[rgb_string] = bin

    # with open('../probabilities/rgb_to_bins.pickle', 'wb') as fp:
    #     pickle.dump(rgb_to_bin, fp)


def bin_probability():
    '''
    :return: Get the probability of colour for each bin accross the whole dataset
    '''
    with open('../probabilities/probability_object_all.pickle', 'rb') as fp:
        probabilities = pickle.load(fp)

    with open('../probabilities/rgb_to_bins.pickle', 'rb') as fp:
        rgb_to_bin = pickle.load(fp)

    bin_probs = np.zeros(262)

    for r in range(256):
        print('R value', r)
        for g in range(256):
            for b in range(256):
                key = str(r) + '.0, ' + str(g) + '.0, ' + str(b) + '.0'
                bin = rgb_to_bin[key]
                bin_probs[bin] += probabilities.absolutes[r, g, b]

    bin_probs /= probabilities.counts

    with open('../probabilities/probability_bins.pickle', 'wb') as fp:
        pickle.dump(bin_probs, fp)

def compute_weights():
    with open('../probabilities/probability_bins.pickle', 'rb') as fp:
        bin_probs = pickle.load(fp)

    print('Number of bins', len(bin_probs))
    weights = {}
    somme = 0
    for key in range(len(bin_probs)):
        prob = bin_probs[key]
        w = probs_to_weight(prob, len(bin_probs))
        weights[key] = w
        somme += w * prob

    print(somme)

    newSomme = 0
    for key in range(len(bin_probs)):
        weights[key] = weights[key] / somme
        newSomme += weights[key] * bin_probs[key]
    print(newSomme)

    with open('../probabilities/weights2.pickle', 'wb') as fp:
        pickle.dump(weights, fp)

    waitlist = []
    for key in range(len(bin_probs)):
        waitlist.append(weights[key])

    with open('../probabilities/waitlist2.pickle', 'wb') as fp:
        pickle.dump(waitlist, fp)


def probs_to_weight(weight, Q, sigma=5, lamda=0.5):
    # smoothed = weight
    gauss = norm(scale=np.sqrt(sigma))
    smoothed = gauss.pdf(weight)
    smoothed = smoothed / sigma
    smoothed = (1 - lamda) * smoothed + (lamda / Q)
    smoothed = 1. / smoothed

    return smoothed

def test_weight_loss():
    image = read_image('../test_images/fish.JPEG')
    image = convert_rgb_to_lab(image)
    image = soft_encode_lab_img(image)
    print(image.shape)

    with open('../probabilities/weights.pickle', 'rb') as fp:
        weights = pickle.load(fp)
    print("Shape", image.shape)

    test = np.random.rand(64, 64, 262)
    losses = 0
    for h in range(image.shape[0]):
        loss = np.dot(image[h],
                            np.log(test[h] + 0.000000000000000001).transpose())
        loss = np.diag(loss)
        loss = - loss
        loss = np.sum(loss)
        losses += loss
    print(losses)

    losses = 0
    for h in range(test.shape[0]):
        vs = np.array([weights[np.argmax(x)] for x in image[h]])
        loss = vs[:, np.newaxis] * np.dot(image[h],
                                          np.log(test[h] + 0.000000000000000001).transpose())
        loss = np.diag(loss)
        loss = - loss
        loss = np.sum(loss)
        losses += loss
    print(losses)

    return losses


def main():
    compute_weights()
    with open('../probabilities/weights2.pickle', 'rb') as fp:
        weights = pickle.load(fp)

    x = []
    y = []
    for i in range(len(weights)):
        x.append(i)
        y.append(weights[i])

    plt.scatter(x, y, color='teal', s=0.3)
    plt.savefig('../ResultPics/weights2.png')
    plt.show()

if __name__ == '__main__':
    main()
# bin_probability()


