#!/usr/bin/env bash

for i in 0 1 42
do

    python -m code.train -c ./config/qqp/10_lbl_5000_unlbl.json -k transform_type=SynonymReplacement seed=$i emb_aug=adv

    python -m code.train -c ./config/qqp/100_lbl_5000_unlbl.json -k transform_type=SynonymReplacement seed=$i emb_aug=adv

done