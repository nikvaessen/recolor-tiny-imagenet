import numpy as np

from keras import Sequential
from keras import callbacks

from recolor.keras_util import DataGenerator, OutputProgress

import recolor.constants as c

from recolor.cic_paper_network import init_model, multinomial_loss, required_input_shape_


def train_model_small_dataset_multinomial_loss():
    # define sensible default parameters
    params = {
        'dim_in': (64, 64, 1),
        'dim_out': (64, 64, c.num_lab_bins),
        'batch_size': 8,
        'shuffle': True,
        'mode': DataGenerator.mode_grey_in_softencode_out
    }

    # define data generators
    train_partition = c.training_set_tiny_file_paths
    validation_partition = c.validation_set_tiny_file_paths

    train_partition = train_partition[0:4*8]
    validation_partition = validation_partition[0:4*8]

    training_generator = DataGenerator(train_partition, **params)
    validation_generator = DataGenerator(validation_partition, **params)

    # model.summary()
    model: Sequential = init_model(loss_function=multinomial_loss,
                                   batch_size=params['batch_size'])

    tb_callback = callbacks.TensorBoard(log_dir='../tensorboard',
                                        histogram_freq=0,
                                        write_graph=True,
                                        write_images=True)

    lr_callback = callbacks.ReduceLROnPlateau()

    op_callback = OutputProgress(train_partition[5:8], required_input_shape_,
                                 "../tensorboard/")

    save_callback = callbacks.ModelCheckpoint("weights.{epoch:02d}-{val_loss:.2f}.hdf5", period=2)

    # To use with model generator/
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=4,
                        verbose=1,
                        epochs=10,
                        callbacks=[tb_callback, lr_callback, op_callback, save_callback])

    # take a sample and try to predict
    from recolor import image_util

    # Image 1
    sample_rgb = image_util.read_image(train_partition[18])
    sample_lab = image_util.convert_rgb_to_lab(sample_rgb)

    sample_grey = sample_lab[:, :, 0:1]
    sample_grey = sample_grey.reshape(1, *sample_grey.shape)

    output = model.predict(sample_grey)

    ab = image_util.probability_dist_to_ab(output)

    lab = np.zeros(sample_lab.shape)
    lab[:, :, 0:1] = sample_grey
    lab[:, :, 1:] = ab

    rgb = image_util.convert_lab_to_rgb(lab)

    image_util.plot_img_converted(sample_rgb, lambda x: rgb)

    # Image 2

    sample_rgb = image_util.read_image(train_partition[30])
    sample_lab = image_util.convert_rgb_to_lab(sample_rgb)

    sample_grey = sample_lab[:, :, 0:1]
    sample_grey = sample_grey.reshape(1, *sample_grey.shape)

    output = model.predict(sample_grey)

    ab = image_util.probability_dist_to_ab(output)

    lab = np.zeros(sample_lab.shape)
    lab[:, :, 0:1] = sample_grey
    lab[:, :, 1:] = ab

    rgb = image_util.convert_lab_to_rgb(lab)

    image_util.plot_img_converted(sample_rgb, lambda x: rgb)


def main():
    train_model_small_dataset_multinomial_loss()


if __name__ == '__main__':
    main()