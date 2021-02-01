#!/usr/bin/env bash


python -m code.normal_train -c ./config/mnli/10_lbl_0_unlbl.json -k transform_type=BackTranslation seed=0

python -m code.normal_train -c ./config/mnli/10_lbl_0_unlbl.json -k transform_type=BackTranslation seed=1

python -m code.normal_train -c ./config/mnli/10_lbl_0_unlbl.json -k transform_type=BackTranslation seed=42

python -m code.normal_train -c ./config/mnli/100_lbl_0_unlbl.json -k transform_type=BackTranslation seed=0

python -m code.normal_train -c ./config/mnli/100_lbl_0_unlbl.json -k transform_type=BackTranslation seed=1

python -m code.normal_train -c ./config/mnli/100_lbl_0_unlbl.json -k transform_type=BackTranslation seed=42

python -m code.normal_train -c ./config/mnli/100_lbl_0_unlbl.json -k transform_type=WordReplacementLM seed=42






