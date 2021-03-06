################################################################################
# Will run each an experiment given by a 'yaml' experiment configuration
#
# Author(s): Nik Vaessen, Jade Cock
################################################################################

import sys
import os
import yaml
import json
import time
import subprocess

import keras

from recolor.keras_util import TrainingConfig, train

################################################################################

models_subfolder = 'models'
tensorboard_subfolder = 'tensorboard-log-dir'
progression_subfolder = 'progression'
subfolders = [models_subfolder, tensorboard_subfolder, progression_subfolder]


def create_result_dir(yaml_config_dic, storage_dir_path, restart=False):
    name = yaml_config_dic['name']

    experiment_path = os.path.join(storage_dir_path, name)

    if not os.path.isdir(experiment_path):
        os.mkdir(experiment_path)

    index = 0
    while True:
        if index > 100:
            print("please do not run so many experiments on the same machine!")
            exit()

        experiment_path_subfolder = os.path.join(experiment_path, "run{}".format(index))

        if os.path.isdir(experiment_path_subfolder):
            index += 1
        else:
            os.mkdir(experiment_path_subfolder)
            for sf in subfolders:
                os.mkdir(os.path.join(experiment_path_subfolder, sf))
            return experiment_path_subfolder


def get_training_config(yaml_config, storage_path) -> TrainingConfig:
    model = yaml_config['use_network']

    # training section
    training = yaml_config['training']
    
    dim_in = training[0]['dim_in']
    dim_out = training[1]['dim_out']
    n_epochs = training[2]['n_epochs']
    n_workers = training[3]['n_workers']
    queue_size = training[4]['queue_size']
    batch_size = training[5]['batch_size']
    shuffle = training[6]['shuffle']

    mode = training[7]['mode']
    dataset = training[8]['dataset']
    loss = training[9]['loss']

    # callback obj
    callbacks = yaml_config['callbacks']
    
    use_tensorboard = callbacks[0]['tensorboard'][0]['use_tensorboard']
    tensorboard_log_dir = os.path.join(storage_path, tensorboard_subfolder)

    reduce_lr_on_plateau = callbacks[1]['reducing-learning-rate'][0]['reduce_lr_on_plateau']
    reduce_lr_on_plateau_factor = callbacks[1]['reducing-learning-rate'][1]['factor']
    reduce_lr_on_plateau_patience = callbacks[1]['reducing-learning-rate'][2]['patience']
    reduce_lr_on_plateau_cooldown = callbacks[1]['reducing-learning-rate'][3]['cooldown']
    reduce_lr_on_plateau_delta = callbacks[1]['reducing-learning-rate'][4]['delta']

    save = yaml_config['callbacks'][2]['save']

    save_colored_image_progress = save[0]['colorisation-progress'][0]['save_colorisation']
    image_paths_to_save = save[0]['colorisation-progress'][1]["path_to_colorisation_images"]
    image_paths_to_save = os.path.abspath(image_paths_to_save)
    image_progression_log_dir = os.path.join(storage_path, progression_subfolder)
    image_progression_period = save[0]['colorisation-progress'][2]["progression_period"]

    periodically_save_model = save[1]['periodically-save-model'][0]['save_periodically']
    periodically_save_model_period = save[1]['periodically-save-model'][1]['psm_period']
    periodically_save_model_fn = save[1]['periodically-save-model'][2]['psm_file_name']
    periodically_save_model_path = os.path.join(storage_path,
                                                models_subfolder,
                                                periodically_save_model_fn)

    save_best_model = save[2]['save-best-model'][0]['save_best']
    save_best_model_fn = save[2]['save-best-model'][1]['sbm_file_name']
    save_best_model_path = os.path.join(storage_path,
                                        models_subfolder,
                                        save_best_model_fn)

    # print("dim_in", dim_in)
    # print("dim_out", dim_out)
    # print("n_epochs", n_epochs)
    # print("n_workers", n_workers)
    # print("batch_size", batch_size)
    # print("shuffle", shuffle)
    # print("mode", mode)
    # print("dataset", dataset)
    # print("loss", loss)
    # print("use_tensorboard", use_tensorboard)
    # print("tensorboard_log_dir", tensorboard_log_dir)
    # print("reduce_lr_on_plateau", reduce_lr_on_plateau)
    # print("save_colored_image_progress", save_colored_image_progress)
    # print("image_paths_to_save", image_paths_to_save)
    # print("image_progression_log_dir", image_progression_log_dir)
    # print("image_progression_period", image_progression_period)
    # print("periodically_save_model", periodically_save_model)
    # print("periodically_save_model_path", periodically_save_model_path)
    # print("periodically_save_model_period", periodically_save_model_period)
    # print("save_best_model", save_best_model)
    # print("save_best_model_path", save_best_model_path)
    #
    config = TrainingConfig(
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
        save_best_model_path
    )

    return config


def execute_config(config_path, storage_path, restart_model=None):
    print("Reading config from", config_path)

    config_path = os.path.abspath(config_path)

    if not os.path.isfile(config_path):
        print("invalid path to config file")
        exit()

    with open(config_path, 'r') as fp:
        yaml_config = yaml.safe_load(fp)
        # print(json.dumps(yaml_config, indent=4))

    storage_path = create_result_dir(yaml_config, storage_path)

    print("Storing results in", storage_path)

    with open(os.path.join(storage_path, "config.json"), 'w') as f:
        json.dump(yaml_config, f, indent=4)

    training_config = get_training_config(yaml_config, storage_path)

    start_training_timestamp = time.time()
    model = training_config.get_init_model(restart_model=restart_model)
    train(model, training_config)
    end_training_timestamp = time.time()

    time_elapsed = end_training_timestamp - start_training_timestamp

    print("training took", time_elapsed, "seconds")


def main():
    n_args = len(sys.argv)
    if n_args != 3 and n_args != 4:
        print("Usage: python3 run_experiment /path/to/queue_folder/ "
              "/path/to/where/results/are/stored <path/to/restart/model>")
        exit()

    queue_path = os.path.abspath(sys.argv[1])

    if not os.path.isdir(queue_path):
        print("given queue path {} is not a directory".format(queue_path))
        exit()

    storage_path = os.path.abspath(sys.argv[2])
    if not os.path.isdir(storage_path):
        print('given storage path {} is not a directory'.format(storage_path))
        exit()

    if n_args == 4:
        model_path = os.path.abspath(sys.argv[3])
    else:
        model_path = None

    for file in os.listdir(queue_path):
        if "yaml" in file:
            config_path = os.path.join(queue_path, file)
            execute_config(config_path, storage_path, model_path)

    subprocess.call('../upload.sh')

    time.sleep(3)

    subprocess.call('sudo shutdown -h now'.split(" "))


if __name__ == '__main__':
    main()
