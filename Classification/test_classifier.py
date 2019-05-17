import pickle
import keras

import data_utils

model_path = "vgg_classification/run1/bestmodels/best_model.hdf5"
test_ids_fn = "test_ids_tiny.pickle"


def main():
    with open(test_ids_fn, 'rb') as f:
        test_file_paths = pickle.load(f)
        print(test_file_paths[0])

    with open('id_label.pickle', 'rb') as f:
        labels = pickle.load(f)
        print(labels)

    with open('id_name.pickle', 'rb') as f:
        name_to_label = pickle.load(f)
        print(name_to_label)

    with open('label_id.pickle', 'rb') as f:
        lable_id = pickle.load(f)
        print(lable_id)

    #
    # mode = 'colour'
    # dim_in = (64, 64, 3)
    # shuffle = True
    # batch_size = 1
    # n_classes = 30
    # dim_out = (n_classes)
    #
    # training_generator = data_utils.DataGenerator(test_file_paths, batch_size,
    #                                               dim_in, dim_out, shuffle,
    #                                               mode, 'validation')
    #
    # num_samples = training_generator.__len__()
    #
    # sample = training_generator[0]
    #
    # print(sample)

    # model = keras.models.load_model(model_path)
    # model = keras.models.Model()



    pass


if __name__ == '__main__':
    main()
