#!/usr/bin/env bash

for i in 0
do

    python -m code.train -c ./config/mrpc/10_lbl_5000_unlbl.json -k transform_type=SynonymReplacement seed=$i

    python -m code.train -c ./config/mrpc/10_lbl_5000_unlbl.json -k transform_type=WordReplacementVocab seed=$i

    python -m code.train -c ./config/mrpc/10_lbl_5000_unlbl.json -k transform_type=RandomInsertion seed=$i

    python -m code.train -c ./config/mrpc/10_lbl_5000_unlbl.json -k transform_type=RandomDeletion seed=$i

    python -m code.train -c ./config/mrpc/10_lbl_5000_unlbl.json -k transform_type=RandomSwapping seed=$i

    python -m code.train -c ./config/mrpc/10_lbl_5000_unlbl.json -k transform_type=Cutoff seed=$i

    python -m code.train -c ./config/mrpc/10_lbl_5000_unlbl.json -k transform_type=WordReplacementLM seed=$i

    python -m code.train -c ./config/mrpc/10_lbl_5000_unlbl.json -k transform_type=BackTranslation seed=$i

    python -m code.train -c ./config/mrpc/100_lbl_5000_unlbl.json -k transform_type=SynonymReplacement seed=$i

    python -m code.train -c ./config/mrpc/100_lbl_5000_unlbl.json -k transform_type=WordReplacementVocab seed=$i

    python -m code.train -c ./config/mrpc/100_lbl_5000_unlbl.json -k transform_type=RandomInsertion seed=$i

    python -m code.train -c ./config/mrpc/100_lbl_5000_unlbl.json -k transform_type=RandomDeletion seed=$i

    python -m code.train -c ./config/mrpc/100_lbl_5000_unlbl.json -k transform_type=RandomSwapping seed=$i

    python -m code.train -c ./config/mrpc/100_lbl_5000_unlbl.json -k transform_type=Cutoff seed=$i

    python -m code.train -c ./config/mrpc/100_lbl_5000_unlbl.json -k transform_type=WordReplacementLM seed=$i

    python -m code.train -c ./config/mrpc/100_lbl_5000_unlbl.json -k transform_type=BackTranslation seed=$i

done