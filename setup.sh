#!/usr/bin/env bash
set -e

# installing packages
sudo apt-get install -y python3-pip python3-tk tmux python-pip python3-venv zip unzip
pip3 install --user virtualenv

# creating python virtualenv
virtualenv venv -p python3
source venv/bin/activate
pip3 --no-cache-dir install -r requirements.txt

# downloading data
./download_data.sh

# include root folder in PYTHONPATH so that modules can be found
export PYTHONPATH=${PYTHONPATH}:$(pwd)

deactivate
