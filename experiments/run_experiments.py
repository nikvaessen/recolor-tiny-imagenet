################################################################################
# Will run each an experiment given by a 'yaml' experiment configuration
#
# Author(s): Nik Vaessen
################################################################################

import sys
import os
import yaml
import json

from functools import reduce

from recolor.cic_paper_network import TrainingConfig, train

################################################################################

models_subfolder = 'models'
tensorboard_subfolder = 'tensorboard-log-dir'
progression_subfolder = 'progression'
subfolders = [models_subfolder, tensorboard_subfolder, progression_subfolder]


def create_result_dir(yaml_config_dic, storage_dir_path):
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
    # training section
    training = yaml_config['training']
    
    dim_in = training[0]['dim_in']
    dim_out = training[1]['dim_out']
    n_epochs = training[2]['n_epochs']
    n_workers = training[3]['n_workers']
    batch_size = training[4]['batch_size']
    shuffle = training[5]['shuffle']

    mode = training[6]['mode']
    dataset = training[7]['dataset']
    loss = training[8]['loss']

    # callback obj
    callbacks = yaml_config['callbacks']
    
    use_tensorboard = callbacks[0]['tensorboard'][0]['use_tensorboard']
    tensorboard_log_dir = os.path.join(storage_path, tensorboard_subfolder)

    reduce_lr_on_plateau = callbacks[1]['reducing-learning-rate'][0]['reduce_lr_on_plateau']

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

    config = TrainingConfig(
        dim_in,
        dim_out,
        n_epochs,
        n_workers,
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
        save_best_model_path
    )

    return config


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 run_experiment /path/to/experiment_config.yaml /path/to/where/results/are/stored")
        exit()

    config_path = os.path.abspath(sys.argv[1])

    if not os.path.isfile(config_path):
        print("invalid path to config file")
        exit()

    storage_path = os.path.abspath(sys.argv[2])
    if not os.path.isdir(storage_path):
        print('given storage path {} is not a directory'.format(storage_path))
        exit()

    print("Reading config from", config_path)

    with open(config_path, 'r') as fp:
        yaml_config = yaml.safe_load(fp)
        # print(json.dumps(yaml_config, indent=4))

    storage_path = create_result_dir(yaml_config, storage_path)

    print("Storing results in", storage_path)

    training_config = get_training_config(yaml_config, storage_path)

    model = training_config.get_init_model()
    train(model, training_config)

    should_shutdown = yaml_config['shutdown-on-completion']
    if should_shutdown:
        import subprocess
        subprocess.call('sudo shutdown')


if __name__ == '__main__':
    main()
