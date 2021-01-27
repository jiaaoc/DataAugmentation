#!/usr/bin/env bash

for i in 0 1 42
do
    python -m code.normal_train -c ./config/qnli/10_lbl_0_unlbl.json -k transform_type=None seed=$i

    python -m code.normal_train -c ./config/qnli/10_lbl_0_unlbl.json -k transform_type=SynonymReplacement seed=$i

    python -m code.normal_train -c ./config/qnli/10_lbl_0_unlbl.json -k transform_type=WordReplacementVocab seed=$i

    python -m code.normal_train -c ./config/qnli/10_lbl_0_unlbl.json -k transform_type=RandomInsertion seed=$i

    python -m code.normal_train -c ./config/qnli/10_lbl_0_unlbl.json -k transform_type=RandomDeletion seed=$i

    python -m code.normal_train -c ./config/qnli/10_lbl_0_unlbl.json -k transform_type=RandomSwapping seed=$i

    python -m code.normal_train -c ./config/qnli/10_lbl_0_unlbl.json -k transform_type=Cutoff seed=$i
done


for i in 0 1 42
do
    python -m code.normal_train -c ./config/qnli/100_lbl_0_unlbl.json -k transform_type=None seed=$i

    python -m code.normal_train -c ./config/qnli/100_lbl_0_unlbl.json -k transform_type=SynonymReplacement seed=$i

    python -m code.normal_train -c ./config/qnli/100_lbl_0_unlbl.json -k transform_type=WordReplacementVocab seed=$i

    python -m code.normal_train -c ./config/qnli/100_lbl_0_unlbl.json -k transform_type=RandomInsertion seed=$i

    python -m code.normal_train -c ./config/qnli/100_lbl_0_unlbl.json -k transform_type=RandomDeletion seed=$i

    python -m code.normal_train -c ./config/qnli/100_lbl_0_unlbl.json -k transform_type=RandomSwapping seed=$i

    python -m code.normal_train -c ./config/qnli/100_lbl_0_unlbl.json -k transform_type=Cutoff seed=$i
done








