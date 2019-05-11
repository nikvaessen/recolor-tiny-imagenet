################################################################################
# Will run each an experiment given by a 'yaml' experiment configuration
#
# Author(s): Nik Vaessen
################################################################################

import sys
import os
import yaml

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


def find(value, obj):
    result = find_recur(value, obj)

    if result is None:
        raise ValueError("could not find", value)
    else:
        return result


def find_recur(value, obj):
    if isinstance(obj, dict):
        if value in obj:
            return obj[value]
        else:
            result = None
            for key in obj:
                if result is not None:
                    return result
                else:
                    result = find(value, obj[key])
    elif isinstance(obj, list):
        result = None
        for o in obj:
            if result is not None:
                return result
            else:
                return find(value, o)
    else:
        return


def get_training_config(yaml_config, storage_path) -> TrainingConfig:
    dim_in = find('dim_in', yaml_config)
    dim_out = find('dim_out', yaml_config)

    n_epochs = find('n_epochs', yaml_config)
    n_workers = find('n_workers', yaml_config)
    batch_size = find('batch_size', yaml_config)
    shuffle = find('shuffle', yaml_config)

    mode = find('mode', yaml_config)
    dataset = find('dataset', yaml_config)
    loss = find('loss', yaml_config)

    use_tensorboard = find('use_tensorboard', yaml_config)
    tensorboard_log_dir = os.path.join(storage_path, tensorboard_subfolder)

    reduce_lr_on_plateau = find('reduce_lr_on_plateau', yaml_config)

    save_colored_image_progress = find('save_colorisation', yaml_config)
    image_paths_to_save = find('path_to_colorisation_images', yaml_config)
    image_progression_log_dir = os.path.join(storage_path, progression_subfolder)

    periodically_save_model = find('save_periodically', yaml_config)
    periodically_save_model_period = find('psm_period', yaml_config)
    periodically_save_model_path = os.path.join(storage_path,
                                                models_subfolder,
                                                find('psm_file_name', yaml_config))

    save_best_model = find('save_best', yaml_config)
    save_best_model_path = os.path.join(storage_path,
                                        models_subfolder,
                                        find('sbm_file_name', yaml_config))

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
        periodically_save_model,
        periodically_save_model_period,
        periodically_save_model_path,
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
