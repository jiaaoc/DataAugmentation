#!/usr/bin/env bash


python -m code.train -c ./config/qqp/10_lbl_5000_unlbl.json -k transform_type=WordReplacementLM seed=0

python -m code.train -c ./config/qqp/10_lbl_5000_unlbl.json -k transform_type=WordReplacementLM seed=1

python -m code.train -c ./config/qqp/10_lbl_5000_unlbl.json -k transform_type=WordReplacementLM seed=42

python -m code.train -c ./config/qqp/10_lbl_5000_unlbl.json -k transform_type=BackTranslation seed=0

python -m code.train -c ./config/qqp/10_lbl_5000_unlbl.json -k transform_type=BackTranslation seed=1

python -m code.train -c ./config/qqp/10_lbl_5000_unlbl.json -k transform_type=BackTranslation seed=42

python -m code.train -c ./config/qqp/100_lbl_5000_unlbl.json -k transform_type=WordReplacementLM seed=0

python -m code.train -c ./config/qqp/100_lbl_5000_unlbl.json -k transform_type=WordReplacementLM seed=1

python -m code.train -c ./config/qqp/100_lbl_5000_unlbl.json -k transform_type=WordReplacementLM seed=42

python -m code.train -c ./config/qqp/100_lbl_5000_unlbl.json -k transform_type=BackTranslation seed=0

python -m code.train -c ./config/qqp/100_lbl_5000_unlbl.json -k transform_type=BackTranslation seed=1

python -m code.train -c ./config/qqp/100_lbl_5000_unlbl.json -k transform_type=BackTranslation seed=42