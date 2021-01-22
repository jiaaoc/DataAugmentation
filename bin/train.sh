#!/usr/bin/env bash

data_path=$1
batch_size=$2
transform_type=$3
n_labeled=$4
un-labeled=$5

python -m code.train --data-path=$data_path --batch-size=$batch_size --transform-type=$transform_type --n-labeled=$n_labeled --epochs=20 --un-labeled=$un-labeled

