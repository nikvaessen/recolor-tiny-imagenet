#!/usr/bin/env bash

set -e

tmux kill-server

tmux new-session -d -s exp
tmux send -t exp "source venv/bin/activate" ENTER
tmux send -t exp "export PYTHONPATH=${PYTHONPATH}:../" ENTER
tmux send -t exp "python3 run_experiments.py queue/ results/" ENTER

tmux new-session -d -s tensorboard
tmux send -t tensorboard "source .env/bin/activate" ENTER
tmux send -t tensorboard "tensorboard --logdir results/" ENTER

tmux new-session -d -s sync
tmux send -t sync "while true; do ../upload.sh; sleep 600; done" ENTER

