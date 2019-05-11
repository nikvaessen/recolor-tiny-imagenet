#!/usr/bin/env python3

import os
import zipfile
import wget

file_name = 'tiny-imagenet-200.zip'
folder_name = 'tiny-imagenet-200'

url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'

data_dir = 'data'
destination_dir = os.path.join(data_dir, folder_name)

if not os.path.isdir(data_dir):
    os.mkdir(destination_dir)

zip_file_name = os.path.join(data_dir, file_name)

if not os.path.exists(zip_file_name):
    print("downloading data...")
    wget.download(url, out=destination_dir)

if not os.path.exists(destination_dir):
    print("unzipping data file...")
    zip_ref = zipfile.ZipFile(zip_file_name, 'r')
    zip_ref.extractall(data_dir)
    zip_ref.close()
