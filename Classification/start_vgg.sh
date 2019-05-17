#!/usr/bin/env bash

set -e


tmux new-session -d -s tensorboard
tmux send -t tensorboard "source .env/bin/activate" ENTER
tmux send -t tensorboard "tensorboard --logdir vgg_classification" ENTER

source ../venv/bin/activate
export PYTHONPATH=${PYTHONPATH}:../

python3 image_classifier.py

sudo shutdown now

