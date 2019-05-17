import pickle
import keras

import numpy as np

import data_utils
from image_classifier import init_model

model_path = "vgg_classification/run8/bestmodels/best_model.hdf5"
test_ids_fn = "test_ids_tiny.pickle"


def main():
    mode = 'colour'

    if mode == 'gray':
        with open('./validation_ids_gray.pickle', 'rb') as fp:
            validation_paths = pickle.load(fp)

    elif mode == 'recoloured':
        with open('./validation_ids_recolour.pickle', 'rb') as fp:
            validation_paths = pickle.load(fp)

    elif mode == 'colour':
        with open('./validation_ids_tiny.pickle', 'rb') as fp:
            validation_paths = pickle.load(fp)

    dim_in = (64, 64, 3)
    shuffle = True
    batch_size = 32
    n_classes = 30
    dim_out = (n_classes)

    training_generator = data_utils.DataGenerator(validation_paths, batch_size,
                                                  dim_in, dim_out, shuffle,
                                                  mode, 'validation')

    model = keras.models.load_model(model_path)

    total = 0
    correct = 0
    for batch in training_generator:
        X, y = batch
        total += X.shape[0]

        y = np.argmax(y, axis=1)
        p = np.argmax(model.predict(X), axis=1)

        c = (y == p).sum()
        correct += c

    print("acc:", correct/total)


    pass


if __name__ == '__main__':
    main()
