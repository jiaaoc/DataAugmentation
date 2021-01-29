#!/usr/bin/env bash


python -m code.train -c ./config/qnli/10_lbl_5000_unlbl.json -k transform_type=SynonymReplacement seed=1

python -m code.train -c ./config/qnli/10_lbl_5000_unlbl.json -k transform_type=SynonymReplacement seed=42

python -m code.train -c ./config/qnli/100_lbl_5000_unlbl.json -k transform_type=WordReplacementLM seed=42

python -m code.train -c ./config/qnli/100_lbl_5000_unlbl.json -k transform_type=BackTranslation seed=42
