################################################################################
#
# Transfer learning of VGG imagenet weight to do colorization instead.
#
# Author(s): Nik Vaessen
#
################################################################################

import recolor.constants as c

from recolor.networks import init_vgg_transfer_model

################################################################################
# enable training of network


def main():
    from keras_util import DataGenerator

    batch_size = 16
    model = init_vgg_transfer_model((64, 64, 3), batch_size)

    model.summary()

    training_gen = DataGenerator(c.training_set_debug_file_paths, batch_size,
                                 (64, 64, 3),
                                 (64, 64, c.num_lab_bins),
                                 True,
                                 mode=DataGenerator.mode_rgb_in_softencode_out)

    validation_gen = DataGenerator(c.training_set_debug_file_paths, batch_size,
                                   (64, 64, 3),
                                   (64, 64, c.num_lab_bins),
                                   True,
                                   mode=DataGenerator.mode_rgb_in_softencode_out)

    model.fit_generator(generator=training_gen,
                        validation_data=validation_gen,
                        use_multiprocessing=True,
                        workers=3,
                        max_queue_size=3,
                        verbose=1,
                        epochs=2,
                        callbacks=[])


if __name__ == '__main__':
    main()
