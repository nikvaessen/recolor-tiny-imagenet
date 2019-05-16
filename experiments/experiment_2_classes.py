import os

import numpy as np

from recolor.constants import load_pickled_data


prob_dir = os.path.join("..", "probabilities")

weights_dog_path = os.path.join(prob_dir, 'weights_list_dog.pickle')
weights_fish_path = os.path.join(prob_dir, 'weights_list_fish.pickle')


weights_dog = np.array(load_pickled_data(weights_dog_path))
weights_fish = np.array(load_pickled_data(weights_fish_path))

print(weights_dog, weights_dog.shape)

from recolor.constants import weights
print(weights, weights.shape)