#!/usr/bin/env bash

for i in 0 1 42
do
    python -m code.normal_train -c ./config/qnli/10_lbl_0_unlbl.json -k transform_type=ensemble seed=$i

    python -m code.normal_train -c ./config/qnli/100_lbl_0_unlbl.json -k transform_type=ensemble seed=$i

done

