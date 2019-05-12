################################################################################
#
# Define the network as described in Colorful Image Colorization by Zhang et al.
# in order to do accomplish the task of image colorization.
#
# The paper (see "../paper/Colorful_Image_Colorization___zhang_et_al.pdf")
# described the following network setup:
#
################################################################################

import json

from keras import backend as K
from keras import Sequential
from keras.layers import Activation, Conv2D, BatchNormalization, UpSampling2D, ZeroPadding2D
from keras.activations import relu, softmax
from keras import callbacks
from keras import optimizers

from .keras_util import DataGenerator, OutputProgress

from . import constants as c

################################################################################
# Custom loss functions


def multinomial_loss(y_true, y_pred):
    """
    :param y_pred: np.array, dimensions should be (n, h, w, q)
    :param soft_encoded: np.array, dimensions should be (n, h, w, q)
    Make sure all values are between 0 and 1, and that the sum of soft_encoded = 1
    :return: loss
    """

    loss = K.categorical_crossentropy(y_true, y_pred, axis=3) # Cross entropy
    loss = K.sum(loss, axis=1) # Sum over all width vectors
    loss = K.sum(loss, axis=1) # Sum over all height vectors
    loss = K.sum(loss) # Sum over all images in the batch

    return loss


def weighted_multinomial_loss(y_true, y_pred):
    """
    :param y_pred: np.array, dimensions should be (n, h, w, q)
    :param soft_encoded: np.array, dimensions should be (n, h, w, q)
    Make sure all values are between 0 and 1, and that the sum of soft_encoded = 1
    :return: loss
    """

    v = K.argmax(y_true, axis=3)
    v = K.one_hot(v, 262)
    v = v * c.weights
    v = K.sum(v, axis=3)

    loss = K.categorical_crossentropy(y_true, y_pred, axis=3)  # Cross entropy
    loss = K.dot(v, loss)
    loss = K.sum(loss, axis=1)  # Sum over all width vectors
    loss = K.sum(loss, axis=1)  # Sum over all height vectors
    loss = K.sum(loss)  # Sum over all images in the batch

    return loss


################################################################################
# Define the network as a Sequential Keras model

_tiny_imagenet_input_shape_ = (64, 64, 1)
_tiny_imagenet_output_shape = (64, 64, c.num_lab_bins)


def init_model(loss_function=multinomial_loss,
               batch_size=None,
               input_shape=_tiny_imagenet_input_shape_):

    model = Sequential()

    # layer 1: (64x64x1) --> (32x32x64)
    model.add(ZeroPadding2D(input_shape=input_shape, batch_size=batch_size,
                            data_format='channels_last', name="layer1"))
    model.add(Conv2D(filters=64, kernel_size=3, padding="valid", strides=(1, 1)))
    model.add(Activation(relu))
    model.add(ZeroPadding2D())
    model.add(Conv2D(filters=64, kernel_size=3,  strides=(2, 2)))
    model.add(Activation(relu))
    model.add(BatchNormalization())

    # layer 2: (32x32x64) --> (16x16x128)
    model.add(ZeroPadding2D(name="layer2"))
    model.add(Conv2D(filters=128, kernel_size=3, padding="valid", strides=(1, 1)))
    model.add(Activation(relu))
    model.add(ZeroPadding2D())
    model.add(Conv2D(filters=128, kernel_size=3,  strides=(2, 2)))
    model.add(Activation(relu))
    model.add(BatchNormalization())

    # layer 3: (16x16x128) --> (8x8x256)
    model.add(ZeroPadding2D(name="layer3"))
    model.add(Conv2D(filters=256, kernel_size=3, padding="valid", strides=(1, 1)))
    model.add(Activation(relu))
    model.add(ZeroPadding2D())
    model.add(Conv2D(filters=256, kernel_size=3, padding="valid", strides=(1, 1)))
    model.add(Activation(relu))
    model.add(ZeroPadding2D())
    model.add(Conv2D(filters=256, kernel_size=3, strides=(2, 2)))
    model.add(Activation(relu))
    model.add(BatchNormalization())

    # layer 4:  (8x8x256) --> (8x8x512)
    model.add(ZeroPadding2D(name='layer4'))
    model.add(Conv2D(filters=512, kernel_size=3, padding="valid", strides=(1, 1)))
    model.add(Activation(relu))
    model.add(ZeroPadding2D())
    model.add(Conv2D(filters=512, kernel_size=3, padding="valid", strides=(1, 1)))
    model.add(Activation(relu))
    model.add(ZeroPadding2D())
    model.add(Conv2D(filters=512, kernel_size=3, padding="valid", strides=(1, 1)))
    model.add(Activation(relu))
    model.add(BatchNormalization())

    # layer 5: (8x8x512)--> (8x8x512)
    model.add(ZeroPadding2D(padding=(2, 2), name='layer5'))
    model.add(Conv2D(filters=512, kernel_size=3, padding="valid", strides=(1, 1), dilation_rate=(2, 2)))
    model.add(Activation(relu))
    model.add(ZeroPadding2D((2, 2)))
    model.add(Conv2D(filters=512, kernel_size=3, padding="valid", strides=(1, 1), dilation_rate=(2, 2)))
    model.add(Activation(relu))
    model.add(ZeroPadding2D((2, 2)))
    model.add(Conv2D(filters=512, kernel_size=3, padding="valid", strides=(1, 1), dilation_rate=(2, 2)))
    model.add(Activation(relu))
    model.add(BatchNormalization())

    # layer 6: (8x8x512)--> (8x8x512)
    model.add(ZeroPadding2D((2, 2), name='layer6'))
    model.add(Conv2D(filters=512, kernel_size=3, padding="valid", strides=(1, 1), dilation_rate=(2, 2)))
    model.add(Activation(relu))
    model.add(ZeroPadding2D((2, 2)))
    model.add(Conv2D(filters=512, kernel_size=3, padding="valid", strides=(1, 1), dilation_rate=(2, 2)))
    model.add(Activation(relu))
    model.add(ZeroPadding2D((2, 2)))
    model.add(Conv2D(filters=512, kernel_size=3, padding="valid", strides=(1, 1), dilation_rate=(2, 2)))
    model.add(Activation(relu))
    model.add(BatchNormalization())

    # layer 7: (8x8x512)--> (8x8x512)
    model.add(ZeroPadding2D(name='layer7'))
    model.add(Conv2D(filters=512, kernel_size=3, padding="valid", strides=(1, 1)))
    model.add(Activation(relu))
    model.add(ZeroPadding2D())
    model.add(Conv2D(filters=512, kernel_size=3, padding="valid", strides=(1, 1)))
    model.add(Activation(relu))
    model.add(ZeroPadding2D())
    model.add(Conv2D(filters=512, kernel_size=3, padding="valid", strides=(1, 1)))
    model.add(Activation(relu))
    model.add(BatchNormalization())

    # layer 8: (8x8x512)--> (16x15x256)
    model.add(UpSampling2D())
    model.add(ZeroPadding2D(name='layer8'))
    model.add(Conv2D(filters=256, kernel_size=3, strides=(1, 1)))
    model.add(Activation(relu))
    model.add(ZeroPadding2D())
    model.add(Conv2D(filters=256, kernel_size=3, padding="valid", strides=(1, 1)))
    model.add(Activation(relu))
    model.add(ZeroPadding2D())
    model.add(Conv2D(filters=256, kernel_size=3, padding="valid", strides=(1, 1)))
    model.add(Activation('relu'))

    # # layer 9: (16x15x256)--> (32x32x128)
    # model.add(UpSampling2D())
    # model.add(Conv2D(filters=128, kernel_size=3, padding="valid", strides=(1, 1)))
    # model.add(Activation(relu))
    # model.add(Conv2D(filters=128, kernel_size=3, padding="valid", strides=(1, 1)))
    # model.add(Activation(relu))
    # model.add(Conv2D(filters=128, kernel_size=3, padding="valid", strides=(1, 1)))
    # model.add(Activation(relu))
    # model.add(BatchNormalization())

    # output layer: (16x16x256)--> (64x64xn_bins)
    model.add(UpSampling2D((4, 4)))
    model.add(Conv2D(filters=c.num_lab_bins, kernel_size=1, strides=(1, 1)))
    model.add(Activation(softmax))

    adam = optimizers.adam()

    model.compile(loss=loss_function, optimizer=adam)

    return model


################################################################################
# enable training of network

class TrainingConfig:

    modes = [DataGenerator.compressed_mode,
             DataGenerator.mode_grey_in_softencode_out,
             DataGenerator.mode_grey_in_ab_out]

    datasets = [c.n_training_set_tiny_uncompressed,
                c.tiny_imagenet_dataset_full,
                c.tiny_imagenet_dataset_tiny,
                c.debug_dataset]

    losses = [c.multinomial_loss, c.weighted_multinomial_loss]

    def __init__(self,
                 dim_in,
                 dim_out,
                 n_epochs,
                 n_workers,
                 queue_size,
                 batch_size,
                 shuffle,
                 mode,
                 dataset,
                 loss,
                 use_tensorboard,
                 tensorboard_log_dir,
                 reduce_lr_on_plateau,
                 save_colored_image_progress,
                 image_paths_to_save,
                 image_progression_log_dir,
                 image_progression_period,
                 periodically_save_model,
                 periodically_save_model_path,
                 periodically_save_model_period,
                 save_best_model,
                 save_best_model_path):

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_epochs = n_epochs
        self.n_workers = n_workers
        self.queue_size = queue_size
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.mode = self._validate_arg_and_return(mode, TrainingConfig.modes)
        self.dataset = self._validate_arg_and_return(dataset, TrainingConfig.datasets)
        self.loss = self._validate_arg_and_return(loss, TrainingConfig.losses)

        self.use_tensorboard = use_tensorboard
        self.tensorboard_log_dir = tensorboard_log_dir

        self.reduce_lr_on_plateau = reduce_lr_on_plateau

        self.save_colored_image_progress = save_colored_image_progress
        self.image_paths_to_save = image_paths_to_save
        self.image_progression_log_dir = image_progression_log_dir
        self.image_progression_period = image_progression_period

        self.periodically_save_model = periodically_save_model
        self.periodically_save_model_path = periodically_save_model_path
        self.periodically_save_model_period = periodically_save_model_period

        self.save_best_model = save_best_model
        self.save_best_model_path = save_best_model_path

    @staticmethod
    def _validate_arg_and_return(value, possible_values):
        if value in possible_values:
            return value
        else:
            raise ValueError("{} needs to be one of {}"
                             .format(value, possible_values))

    def get_generators(self):
        params = {
            'dim_in': self.dim_in,
            'dim_out': self.dim_out,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'mode': self.mode
        }

        if self.dataset == c.tiny_imagenet_dataset_full:
            training_path = c.training_set_full_file_paths
            validation_path = c.validation_set_full_file_paths
        elif self.dataset == c.tiny_imagenet_dataset_tiny:
            training_path = c.training_set_tiny_file_paths
            validation_path = c.validation_set_tiny_file_paths
        else:
            training_path = c.training_set_debug_file_paths
            validation_path = c.validation_set_debug_file_paths

        print('using dataset:', self.dataset)
        print("using {} training samples".format(len(training_path)))
        print('using {} validation samples'.format(len(validation_path)))

        training_generator = DataGenerator(training_path, **params)
        validation_generator = DataGenerator(validation_path, **params)

        return training_generator, validation_generator

    def get_init_model(self):
        if self.loss == c.multinomial_loss:
            loss = multinomial_loss
        elif self.loss == c.weighted_multinomial_loss:
            loss = weighted_multinomial_loss
        else:
            raise ValueError("could not correct loss")

        model = init_model(loss_function=loss,
                           batch_size=self.batch_size,
                           input_shape=self.dim_in)

        return model


def load_config(path_to_json):
    pass


def save_config(training_config: TrainingConfig):
    pass


def train(model: Sequential, config: TrainingConfig):

    training_generator, validation_generator = config.get_generators()

    callback_list = list()

    if config.use_tensorboard:
        print("using tensorboard")
        tb_callback = callbacks.TensorBoard(log_dir=config.tensorboard_log_dir,
                                            write_graph=False,
                                            update_freq=5000
                                            )
        callback_list.append(tb_callback)

    if config.reduce_lr_on_plateau:
        print("reducing learning rate on plateau")
        lr_callback = callbacks.ReduceLROnPlateau()
        callback_list.append(lr_callback)

    if config.save_colored_image_progress:
        print("saving progression every {} epochs".format(config.image_progression_period))
        op_callback = OutputProgress(config.image_paths_to_save,
                                     config.dim_in,
                                     config.image_progression_log_dir,
                                     every_n_epochs=config.image_progression_period)
        callback_list.append(op_callback)

    if config.periodically_save_model:
        print("saving model every {} epcohs".format(config.periodically_save_model_period))
        p_save_callback = callbacks.ModelCheckpoint(config.periodically_save_model_path,
                                                    period=config.periodically_save_model_period)
        callback_list.append(p_save_callback)

    if config.save_best_model:
        print("saving best model")
        best_save_callback = callbacks.ModelCheckpoint(config.save_best_model_path,
                                                       save_best_only=True)
        callback_list.append(best_save_callback)

    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=config.n_workers,
                        max_queue_size=config.queue_size,
                        verbose=1,
                        epochs=config.n_epochs,
                        callbacks=callback_list)

