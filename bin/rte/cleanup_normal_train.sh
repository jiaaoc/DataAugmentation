#!/usr/bin/env bash

for i in 0 1 42
do

    python -m code.normal_train -c ./config/rte/10_lbl_0_unlbl.json -k transform_type=WordReplacementVocab seed=$i

    python -m code.normal_train -c ./config/rte/10_lbl_0_unlbl.json -k transform_type=BackTranslation seed=$i

    python -m code.normal_train -c ./config/rte/10_lbl_0_unlbl.json -k transform_type=WordReplacementLM seed=$i

done


for i in 0 1 42
do
    python -m code.normal_train -c ./config/rte/100_lbl_0_unlbl.json -k transform_type=WordReplacementVocab seed=$i

    python -m code.normal_train -c ./config/rte/100_lbl_0_unlbl.json -k transform_type=BackTranslation seed=$i

    python -m code.normal_train -c ./config/rte/100_lbl_0_unlbl.json -k transform_type=WordReplacementLM seed=$i

done







