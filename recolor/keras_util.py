################################################################################
# Utility methods related to keras functionality:
# Data generator required to generate the data batches at training time
# Callback for saving progress of image coloring
# TrainingConfig for setting up training of a network
#
# author(s): Jade Cock, Nik Vaessen
################################################################################

import os
import pickle

import numpy as np
import keras

import keras.callbacks as callbacks

if __name__ == '__main__' or __name__ == 'keras_util':
    import image_util
    import constants as c
    from networks import multinomial_loss, weighted_multinomial_loss, init_cic_model, init_vgg_transfer_model
else:
    from . import image_util
    from . import constants as c
    from .networks import multinomial_loss, weighted_multinomial_loss, init_cic_model, init_vgg_transfer_model

################################################################################
# Define different ways of reading the data

def load_compressed_files(image_path):
    """
    Load the compressed file image_path.npz where the attribute input
    is the cielab image, and where the attribute ouput is the soft
    encoding version of the expected colour bin as a
    (width*height*num_bins) np array
    :param image_path: path to the npz file
    :return: the cielab image and the soft encoded image in that order
    """

    compressed = np.load(image_path)
    cielab, soft_encode = compressed['input'], compressed['output']

    return cielab, soft_encode


def load_compressed_files(image_path):
    """
    Load the compressed file image_path.npz where the attribute input
    is the cielab image, and where the attribute ouput is the soft
    encoding version of the expected colour bin as a
    (width*height*num_bins) np array
    :param image_path: path to the npz file
    :return: the cielab image and the soft encoded image in that order
    """

    compressed = np.load(image_path)
    cielab, soft_encode = compressed['input'], compressed['output']

    return cielab[:, :, 0:1], soft_encode


def load_rgb_in_softencode_out(image_path):
    path_split = os.path.split(image_path)
    fn = path_split[1]

    tag = fn.split("_")[0]
    num = fn.split("_")[1]

    rgb_save_path = os.path.join(
        path_split[0],
        "{}_{}_rgb.npz".format(tag, num)
    )
    if os.path.exists(rgb_save_path):
        rgb = np.load(rgb_save_path)['arr_0']
    else:
        rgb = image_util.read_image(image_path)

    filename = os.path.split(image_path)[-1]
    filename = os.path.splitext(filename)[0]
    se_filename = os.path.join(c.soft_encoding_training_and_val_dir,
                               filename + c.soft_encoding_filename_postfix)
    if os.path.exists(se_filename):
        soft_encoding = np.load(se_filename)['arr_0']
    else:
        cielab = image_util.convert_rgb_to_lab(rgb)
        soft_encoding = image_util.soft_encode_lab_img(cielab)

    return rgb, soft_encoding


def load_image_grey_in_softencode_out(image_path):
    """
    Load an image, create the input and output of the network.

    The input is a gray-scale image as a (width*height*1) numpy array
    The output is a soft-encoding of the expected color bin as a
    (width*height*num_bins) numpy array

    :param image_path: the path to the image
    :return: tuple of x and y (input and output)
    """
    filename = os.path.split(image_path)[-1]
    filename = os.path.splitext(filename)[0]
    se_filename = os.path.join(c.soft_encoding_training_and_val_dir,
                               filename + c.soft_encoding_filename_postfix)

    rgb = image_util.read_image(image_path)
    cielab = image_util.convert_rgb_to_lab(rgb)

    gray_channel = cielab[:, :, 0]
    gray_channel = gray_channel[:, :, np.newaxis]

    if os.path.exists(se_filename):
        soft_encoding = np.load(se_filename)['arr_0']
    else:
        soft_encoding = image_util.soft_encode_lab_img(cielab)

    return gray_channel, soft_encoding


def load_image_grey_in_ab_out(image_path):
    """
    Load an image, create the input and output of the network.

    The input is a gray-scale image as a (width*height*1) numpy array
    The output is a soft-encoding of the expected color bin as a
    (width*height*num_bins) numpy array

    :param image_path: the path to the image
    :return: tuple of x and y (input and output)
    """
    rgb = image_util.read_image(image_path)
    cielab = image_util.convert_rgb_to_lab(rgb)

    gray_channel = cielab[:, :, 0]
    gray_channel = gray_channel[:, :, np.newaxis]

    ab_channel = cielab[:, :, 1:]

    return gray_channel, ab_channel


################################################################################
# The DataGenerator class used to offload data i/o to CPU while GPU trains
# network


class DataGenerator(keras.utils.Sequence):
    compressed_mode = 'compressed-mode'
    mode_grey_in_ab_out = 'grey-in-ab-out'
    mode_grey_in_softencode_out = 'grey-in-softencode-out'
    mode_rgb_in_softencode_out = 'rgb-in-softencode-out'

    modes = [compressed_mode, mode_grey_in_ab_out,
             mode_grey_in_softencode_out, mode_rgb_in_softencode_out]

    def __init__(self,
                 data_paths,
                 batch_size,
                 dim_in,
                 dim_out,
                 shuffle,
                 mode):
        '''
        :param data_paths: paths to the image files
        :param batch_size: Size of the batches during training
        :param dim_in: Dimension of the input image (in Cielab)
        :param dim_out: Dimension of the output image (in Cielab)
        :param shuffle: Whether to shuffle the input
        :param mode the mode of the data generator. Decides input and
        output of network
        '''
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.batch_size = batch_size
        self.leftovers = len(data_paths) % batch_size
        self.data_paths = data_paths
        self.shuffle = shuffle

        if mode in DataGenerator.modes:
            if mode == DataGenerator.compressed_mode:
                self.image_load_fn = load_compressed_files
            elif mode == DataGenerator.mode_grey_in_ab_out:
                self.image_load_fn = load_image_grey_in_ab_out
            elif mode == DataGenerator.mode_grey_in_softencode_out:
                self.image_load_fn = load_image_grey_in_softencode_out
            elif mode == DataGenerator.mode_rgb_in_softencode_out:
                self.image_load_fn = load_rgb_in_softencode_out
        else:
            raise ValueError("expected mode to be one of", DataGenerator.modes)

        self.indices = []

        self.on_epoch_end()

    def on_epoch_end(self):
        '''
        If shuffle is set to True: shuffles input at the beginning of each epoch
        :return:
        '''

        if self.shuffle:
            self.indices = np.arange(len(self.data_paths))
            np.random.shuffle(self.indices)
            self.indices = self.indices[:len(self.data_paths) - self.leftovers]
        else:
            self.indices = np.arange(len(self.data_paths) - self.leftovers)

    # list_IDs_temp : Ids from the batch to be generated
    def __data_generation(self, batch_paths):
        # Initialization
        X = np.empty((self.batch_size, *self.dim_in))
        y = np.empty((self.batch_size, *self.dim_out))

        # Generate data
        for i, path in enumerate(batch_paths):
            if os.name == 'nt':
                path = path.replace('\\', '/')  # Activate for Windows

            # Store sample
            # print(path)
            inp, outp = self.image_load_fn(path)
            X[i, ] = inp
            y[i, ] = outp

        return X, y

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        # Generate indices of the batch
        indices = self.indices[
                  index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        batch_paths = [self.data_paths[k] for k in indices]

        # Generate data
        X, y = self.__data_generation(batch_paths)

        return X, y


################################################################################
# Define custom callback(s) for our use-case


class OutputProgress(keras.callbacks.Callback):

    def __init__(self,
                 image_paths_file,
                 input_shape,
                 root_dir,
                 must_convert_pdist=True,
                 every_n_epochs=5):
        super().__init__()

        with open(os.path.abspath(image_paths_file), 'r') as f:
            image_paths = f.readlines()

        self.image_paths = [path.strip() for path in image_paths]
        self.batch = np.empty((len(self.image_paths), *input_shape))

        self.root_dir = root_dir
        self.period = every_n_epochs
        self.epochs_since_last_save = 0
        self.must_convert_pdist = must_convert_pdist

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1

        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            self.save_images(str(epoch + 1))

    def on_train_begin(self, logs=None):
        # create batch and store rgb and grey image
        for idx, path in enumerate(self.image_paths):
            rgb = image_util.read_image(path)

            save_path = "img_{}_epoch_{}.png".format(idx, "0_ground_truth")
            save_path = os.path.join(self.root_dir, save_path)
            image_util.save_image(save_path, rgb)

            grey = image_util.read_image(path, as_gray=True)
            save_path = "img_{}_epoch_{}.png".format(idx, "0_grey")
            save_path = os.path.join(self.root_dir, save_path)
            image_util.save_image(save_path, grey)

            lab = image_util.convert_rgb_to_lab(rgb)
            grey = lab[:, :, 0:1]
            self.batch[idx,] = grey

        self.save_images('0_initial_prediction')

    def on_train_end(self, logs=None):
        self.save_images('0_training_end')

    def save_images(self, epoch_string: str):
        y = self.model.predict(self.batch)

        if self.must_convert_pdist:
            y = image_util.probability_dist_to_ab(y)

        for idx in range(self.batch.shape[0]):
            l = self.batch[idx,]
            ab = y[idx,]

            lab = np.empty((l.shape[0], l.shape[1], 3))

            lab[:, :, 0:] = l
            lab[:, :, 1:] = ab

            rgb = (image_util.convert_lab_to_rgb(lab) * 255).astype(np.uint8)
            path = os.path.join(self.root_dir,
                                "img_{}_epoch_{}.png".format(idx, epoch_string))
            image_util.save_image(path, rgb)


################################################################################
# enable training of network

class TrainingConfig:
    cic_model = 'cic_paper_network'
    vgg_model = 'vgg_transfer_network'

    models = [cic_model, vgg_model]



    modes = [DataGenerator.compressed_mode,
             DataGenerator.mode_grey_in_softencode_out,
             DataGenerator.mode_grey_in_ab_out,
             DataGenerator.mode_rgb_in_softencode_out]

    datasets = [c.n_training_set_tiny_uncompressed,
                c.tiny_imagenet_dataset_full,
                c.tiny_imagenet_dataset_tiny,
                c.debug_dataset]

    losses = [c.multinomial_loss, c.weighted_multinomial_loss]

    def __init__(self,
                 model,
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
                 reduce_lr_on_plateau_factor,
                 reduce_lr_on_plateau_patience,
                 reduce_lr_on_plateau_cooldown,
                 reduce_lr_on_plateau_delta,
                 save_colored_image_progress,
                 image_paths_to_save,
                 image_progression_log_dir,
                 image_progression_period,
                 periodically_save_model,
                 periodically_save_model_path,
                 periodically_save_model_period,
                 save_best_model,
                 save_best_model_path):


        self.mode = self._validate_arg_and_return(mode, TrainingConfig.modes)
        self.dataset = self._validate_arg_and_return(dataset, TrainingConfig.datasets)
        self.loss = self._validate_arg_and_return(loss, TrainingConfig.losses)

        self.use_tensorboard = use_tensorboard
        self.tensorboard_log_dir = tensorboard_log_dir

        self.reduce_lr_on_plateau = reduce_lr_on_plateau
        self.reduce_lr_on_plateau_factor = reduce_lr_on_plateau_factor
        self.reduce_lr_on_plateau_patience = reduce_lr_on_plateau_patience
        self.reduce_lr_on_plateau_cooldown = reduce_lr_on_plateau_cooldown
        self.reduce_lr_on_plateau_delta = reduce_lr_on_plateau_delta

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

    def get_init_model(self, restart_model=None):
        if self.loss == c.multinomial_loss:
            loss = multinomial_loss
        elif self.loss == c.weighted_multinomial_loss:
            loss = weighted_multinomial_loss
        else:
            raise ValueError("could not set correct loss")

        if self.model == TrainingConfig.cic_model:
            init_model = init_cic_model
        elif self.model == TrainingConfig.vgg_model:
            init_model = init_vgg_transfer_model
        else:
            raise ValueError("could not set correct model")

        model = init_model(loss_function=loss,
                           batch_size=self.batch_size,
                           input_shape=self.dim_in)

        if restart_model is not None:
            model.load_weights(restart_model)

        return model


def train(model: keras.models.Model, config: TrainingConfig):
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
        lr_callback = callbacks.ReduceLROnPlateau(
            factor=config.reduce_lr_on_plateau_factor,
            patience=config.reduce_lr_on_plateau_patience,
            cooldown=config.reduce_lr_on_plateau_cooldown,
            min_delta=config.reduce_lr_on_plateau_delta
        )
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
