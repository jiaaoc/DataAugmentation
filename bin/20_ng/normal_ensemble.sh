#!/usr/bin/env bash

for i in 0 1 42
do

    python -m code.normal_train -c ./config/20_ng/10_lbl_0_unlbl.json -k transform_type=ensemble seed=$i

    python -m code.normal_train -c ./config/20_ng/100_lbl_0_unlbl.json -k transform_type=ensemble emb_aug=adv seed=$i

done