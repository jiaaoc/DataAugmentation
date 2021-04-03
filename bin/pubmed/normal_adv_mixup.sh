#!/usr/bin/env bash

for i in 0 1 42
do
    python -m code.normal_train -c ./config/pubmed/10_lbl_0_unlbl.json -k transform_type=None emb_aug=adv seed=$i

    python -m code.normal_train -c ./config/pubmed/10_lbl_0_unlbl.json -k transform_type=None emb_aug=mixup seed=$i

    python -m code.normal_train -c ./config/pubmed/100_lbl_0_unlbl.json -k transform_type=None emb_aug=adv seed=$i

    python -m code.normal_train -c ./config/pubmed/100_lbl_0_unlbl.json -k transform_type=None emb_aug=mixup seed=$i

done

