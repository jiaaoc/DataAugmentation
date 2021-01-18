#!/usr/bin/env bash

data_path=$1
batch_size=$2
transform_type=$3
n_labeled=$4

python -m code.normal_train --data-path=$data_path --batch-size=$batch_size --transform-type=$transform_type --n-labeled=$n_labeled --epochs=10

