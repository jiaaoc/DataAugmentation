#!/usr/bin/env bash


python -m code.train -c ./config/pubmed/100_lbl_5000_unlbl.json -k transform_type=WordReplacementLM seed=0

python -m code.train -c ./config/pubmed/100_lbl_5000_unlbl.json -k transform_type=Cutoff seed=0

python -m code.train -c ./config/pubmed/100_lbl_5000_unlbl.json -k transform_type=BackTranslation seed=0

python -m code.train -c ./config/pubmed/100_lbl_5000_unlbl.json -k transform_type=BackTranslation seed=42