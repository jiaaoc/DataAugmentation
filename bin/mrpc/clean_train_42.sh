#!/usr/bin/env bash

for i in 42
do

    python -m code.train -c ./config/mrpc/10_lbl_5000_unlbl.json -k transform_type=RandomSwapping seed=$i

    python -m code.train -c ./config/mrpc/10_lbl_5000_unlbl.json -k transform_type=BackTranslation seed=$i

    python -m code.train -c ./config/mrpc/10_lbl_5000_unlbl.json -k transform_type=WordReplacementLM seed=$i

    python -m code.train -c ./config/mrpc/100_lbl_5000_unlbl.json -k transform_type=RandomSwapping seed=$i

    python -m code.train -c ./config/mrpc/100_lbl_5000_unlbl.json -k transform_type=BackTranslation seed=$i

    python -m code.train -c ./config/mrpc/100_lbl_5000_unlbl.json -k transform_type=WordReplacementLM seed=$i

done