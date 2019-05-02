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

class ColourProbability():
    def __init__(self, num_label, label):
        self.num_label = num_label
        self.label = label
        # if colours is (R:25, G:45, B:65) -> absolutes[25, 45, 65]
        self.absolutes = np.zeros((256, 256, 256))  # Absolute counts of pixels
        self.probabilities = np.zeros((256, 256, 256))  # Probabilities of pixels
        self.counts = 0  # Number of pixels that have been accounted for so far

    def create_probabilities(self):
        self.probabilities /= self.counts

    def add_image(self, dim_x, dim_y, image):
        for i in range(dim_x):
            for j in range(dim_y):
                r = image[i, j, 0]
                g = image[i, j, 1]
                b = image[i, j, 2]
                self._add_rgb(r, g, b)

    def _add_rgb(self, r, g, b):
        self.absolutes[r, g, b] += 1
        self.probabilities[r, g, b] += 1
        self.counts += 1


def test():
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
    label_path = "../Dataset/tiny-imagenet-200/words.txt"
    keys = {}
    with open(label_path) as f:
        for line in f:
            key, val = line.split('\t')
            keys[key] = val
    return keys


def compute_probabilities():
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


compute_probabilities()

