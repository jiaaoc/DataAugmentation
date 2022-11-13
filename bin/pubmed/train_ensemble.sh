#!/usr/bin/env bash

for i in 0 1 42
do

    python -m code.train -c ./config/pubmed/10_lbl_5000_unlbl.json -k transform_type=ensemble seed=$i

    python -m code.train -c ./config/pubmed/100_lbl_5000_unlbl.json -k transform_type=ensemble seed=$i

done