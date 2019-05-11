#!/usr/bin/env bash

FILE=tiny-imagenet-200.zip
URL=http://cs231n.stanford.edu/tiny-imagenet-200.zip

mkdir -p data/
cd data

if [[ ! -f ${FILE} ]]; then
    wget ${URL}
    unzip ${FILE}
fi
