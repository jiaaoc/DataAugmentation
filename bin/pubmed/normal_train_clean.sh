#!/usr/bin/env bash

for i in 0 1 42
do
    python -m code.normal_train -c ./config/pubmed/10_lbl_0_unlbl.json -k transform_type=BackTranslation seed=$i
done


for i in 0 1 42
do

    python -m code.normal_train -c ./config/pubmed/100_lbl_0_unlbl.json -k transform_type=WordReplacementLM seed=$i

    python -m code.normal_train -c ./config/pubmed/100_lbl_0_unlbl.json -k transform_type=BackTranslation seed=$i

done


