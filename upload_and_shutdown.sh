#!/usr/bin/env bash

gsutil -m cp -nr results/ gs://kth-dd2424-bucket/results/
sudo shutdown now