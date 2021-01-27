#!/usr/bin/env bash

data_path=$1
batch_size=$2
batch_size_u=$3
grad_accum_factor=$4
transform_type=$5
n_labeled=$6

python -m code.train --data-path=$data_path --batch-size=$batch_size --batch-size-u=$batch_size_u --grad-accum-factor=$grad_accum_factor  --transform-type=$transform_type --n-labeled=$n_labeled --epochs=10
