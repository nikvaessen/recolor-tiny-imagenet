#!/usr/bin/env bash

TINY_FILE=tiny-imagenet-200.zip
TINY_URL=http://cs231n.stanford.edu/tiny-imagenet-200.zip

mkdir -p data/
cd data

if [[ ! -f ${TINY_FILE} ]]; then
    wget ${TINY_URL}
    unzip ${TINY_FILE}
fi

SOFT_ENCODE_TRAIN_VAL=soft_encoded.zip
TRAIN_VAL_URL=https://storage.googleapis.com/kth-dd2424-bucket/soft_encoded.zip

if [[ ! -f ${SOFT_ENCODE_TRAIN_VAL} ]]; then
    wget ${TRAIN_VAL_URL}
    unzip ${SOFT_ENCODE_TRAIN_VAL}
fi


SOFT_ENCODE_TEST=test.zip
TEST_URL=https://storage.googleapis.com/kth-dd2424-bucket/test.zip

if [[ ! -f ${SOFT_ENCODE_TEST} ]]; then
    wget ${TEST_URL}
    unzip ${SOFT_ENCODE_TEST}
fi