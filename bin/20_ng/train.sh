#!/usr/bin/env bash

for i in 42
do

    CUDA_VISIBLE_DEVICES=7 python -m code.train -c ./config/ag_news/10_lbl_5000_unlbl.json -k transform_type=SynonymReplacement seed=$i

    CUDA_VISIBLE_DEVICES=7 python -m code.train -c ./config/ag_news/10_lbl_5000_unlbl.json -k transform_type=WordReplacementVocab seed=$i

    CUDA_VISIBLE_DEVICES=7 python -m code.train -c ./config/ag_news/10_lbl_5000_unlbl.json -k transform_type=RandomInsertion seed=$i

    CUDA_VISIBLE_DEVICES=7 python -m code.train -c ./config/ag_news/10_lbl_5000_unlbl.json -k transform_type=RandomDeletion seed=$i

    CUDA_VISIBLE_DEVICES=7 python -m code.train -c ./config/ag_news/10_lbl_5000_unlbl.json -k transform_type=RandomSwapping seed=$i

    CUDA_VISIBLE_DEVICES=7 python -m code.train -c ./config/ag_news/10_lbl_5000_unlbl.json -k transform_type=BackTranslation seed=$i

    CUDA_VISIBLE_DEVICES=7 python -m code.train -c ./config/ag_news/10_lbl_5000_unlbl.json -k transform_type=Cutoff seed=$i

done


for i in 42
do

    CUDA_VISIBLE_DEVICES=7 python -m code.train -c ./config/ag_news/100_lbl_5000_unlbl.json -k transform_type=SynonymReplacement seed=$i

    CUDA_VISIBLE_DEVICES=7 python -m code.train -c ./config/ag_news/100_lbl_5000_unlbl.json -k transform_type=WordReplacementVocab seed=$i

    CUDA_VISIBLE_DEVICES=7 python -m code.train -c ./config/ag_news/100_lbl_5000_unlbl.json -k transform_type=RandomInsertion seed=$i

    CUDA_VISIBLE_DEVICES=7 python -m code.train -c ./config/ag_news/100_lbl_5000_unlbl.json -k transform_type=RandomDeletion seed=$i

    CUDA_VISIBLE_DEVICES=7 python -m code.train -c ./config/ag_news/100_lbl_5000_unlbl.json -k transform_type=RandomSwapping seed=$i

    CUDA_VISIBLE_DEVICES=7 python -m code.train -c ./config/ag_news/100_lbl_5000_unlbl.json -k transform_type=BackTranslation seed=$i

    CUDA_VISIBLE_DEVICES=7 python -m code.train -c ./config/ag_news/100_lbl_5000_unlbl.json -k transform_type=Cutoff seed=$i

done




for i in 0 1
do

    CUDA_VISIBLE_DEVICES=7 python -m code.train -c ./config/ag_news/10_lbl_5000_unlbl.json -k transform_type=SynonymReplacement seed=$i

    CUDA_VISIBLE_DEVICES=7 python -m code.train -c ./config/ag_news/10_lbl_5000_unlbl.json -k transform_type=WordReplacementVocab seed=$i

    CUDA_VISIBLE_DEVICES=7 python -m code.train -c ./config/ag_news/10_lbl_5000_unlbl.json -k transform_type=RandomInsertion seed=$i

    CUDA_VISIBLE_DEVICES=7 python -m code.train -c ./config/ag_news/10_lbl_5000_unlbl.json -k transform_type=RandomDeletion seed=$i

    CUDA_VISIBLE_DEVICES=7 python -m code.train -c ./config/ag_news/10_lbl_5000_unlbl.json -k transform_type=RandomSwapping seed=$i

    CUDA_VISIBLE_DEVICES=7 python -m code.train -c ./config/ag_news/10_lbl_5000_unlbl.json -k transform_type=BackTranslation seed=$i

    CUDA_VISIBLE_DEVICES=7 python -m code.train -c ./config/ag_news/10_lbl_5000_unlbl.json -k transform_type=Cutoff seed=$i

done




for i in 0 1
do

    CUDA_VISIBLE_DEVICES=7 python -m code.train -c ./config/ag_news/100_lbl_5000_unlbl.json -k transform_type=SynonymReplacement seed=$i

    CUDA_VISIBLE_DEVICES=7 python -m code.train -c ./config/ag_news/100_lbl_5000_unlbl.json -k transform_type=WordReplacementVocab seed=$i

    CUDA_VISIBLE_DEVICES=7 python -m code.train -c ./config/ag_news/100_lbl_5000_unlbl.json -k transform_type=RandomInsertion seed=$i

    CUDA_VISIBLE_DEVICES=7 python -m code.train -c ./config/ag_news/100_lbl_5000_unlbl.json -k transform_type=RandomDeletion seed=$i

    CUDA_VISIBLE_DEVICES=7 python -m code.train -c ./config/ag_news/100_lbl_5000_unlbl.json -k transform_type=RandomSwapping seed=$i

    CUDA_VISIBLE_DEVICES=7 python -m code.train -c ./config/ag_news/100_lbl_5000_unlbl.json -k transform_type=BackTranslation seed=$i

    CUDA_VISIBLE_DEVICES=7 python -m code.train -c ./config/ag_news/100_lbl_5000_unlbl.json -k transform_type=Cutoff seed=$i

done
