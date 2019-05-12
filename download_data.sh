#!/usr/bin/env bash

TINY_FILE=tiny-imagenet-200.zip
TINY_URL=http://cs231n.stanford.edu/tiny-imagenet-200.zip

mkdir -p data/
cd data

if [[ ! -f ${TINY_FILE} ]]; then
    wget ${TINY_URL}
    unzip ${TINY_FILE}
fi

mkdir -p npz-tiny-imagenet
cd npz-tiny-imagenet

prefix=gs://kth-dd2424-bucket/

ENCODED_TRAIN=encoded_train.zip
ENCODED_TRAIN_URL=${prefix}${ENCODED_TRAIN}

if [[ ! -f ${ENCODED_TRAIN} ]]; then
    gsutil cp -n ${ENCODED_TRAIN_URL} .
    unzip ${ENCODED_TRAIN}
fi

ENCODED_VAL=encoded_val.zip
ENCODED_VAL_URL=${prefix}${ENCODED_VAL}

if [[ ! -f ${ENCODED_VAL} ]]; then
    gsutil cp -n ${ENCODED_VAL_URL} .
    unzip ${ENCODED_VAL}
fi