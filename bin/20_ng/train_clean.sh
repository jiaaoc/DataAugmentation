#!/usr/bin/env bash


python -m code.train -c ./config/20_ng/10_lbl_5000_unlbl.json -k transform_type=WordReplacementLM seed=0

python -m code.train -c ./config/20_ng/10_lbl_5000_unlbl.json -k transform_type=WordReplacementLM seed=1

python -m code.train -c ./config/20_ng/10_lbl_5000_unlbl.json -k transform_type=WordReplacementLM seed=42

python -m code.train -c ./config/20_ng/100_lbl_5000_unlbl.json -k transform_type=WordReplacementLM seed=0

python -m code.train -c ./config/20_ng/100_lbl_5000_unlbl.json -k transform_type=WordReplacementLM seed=1

python -m code.train -c ./config/20_ng/100_lbl_5000_unlbl.json -k transform_type=WordReplacementLM seed=42

python -m code.train -c ./config/20_ng/100_lbl_5000_unlbl.json -k transform_type=WordReplacementVocab seed=0

python -m code.train -c ./config/20_ng/100_lbl_5000_unlbl.json -k transform_type=WordReplacementVocab seed=1

python -m code.train -c ./config/20_ng/100_lbl_5000_unlbl.json -k transform_type=WordReplacementVocab seed=42

python -m code.train -c ./config/20_ng/100_lbl_5000_unlbl.json -k transform_type=WordReplacementVocab seed=1

python -m code.train -c ./config/20_ng/100_lbl_5000_unlbl.json -k transform_type=WordReplacementVocab seed=42
