#!/usr/bin/env bash

data_path=$1
batch_size=$2
batch_size_u=$3
transform_type=$4
n_labeled=$5
gpu=${6-0}

python -m code.train --data-path=$data_path --batch-size=$batch_size --batch-size-u=$batch_size_u --transform-type=$transform_type --n-labeled=$n_labeled --epochs=10 --gpu-$gpu
