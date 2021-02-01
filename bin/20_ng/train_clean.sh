#!/usr/bin/env bash



python -m code.train -c ./config/20_ng/10_lbl_5000_unlbl.json -k transform_type=WordReplacementVocab seed=0

python -m code.train -c ./config/20_ng/10_lbl_5000_unlbl.json -k transform_type=WordReplacementVocab seed=1

python -m code.train -c ./config/20_ng/10_lbl_5000_unlbl.json -k transform_type=WordReplacementVocab seed=42

