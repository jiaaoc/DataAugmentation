#!/usr/bin/env bash

python -m code.train -c ./config/mnli/100_lbl_5000_unlbl.json -k transform_type=RandomSwapping seed=0

python -m code.train -c ./config/mnli/100_lbl_5000_unlbl.json -k transform_type=RandomSwapping seed=1

python -m code.train -c ./config/mnli/100_lbl_5000_unlbl.json -k transform_type=RandomSwapping seed=42

python -m code.train -c ./config/mnli/100_lbl_5000_unlbl.json -k transform_type=WordReplacementLM seed=42
