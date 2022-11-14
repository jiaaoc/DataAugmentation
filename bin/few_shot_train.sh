#!/usr/bin/env bash

python code/normal_train.py -c config/qnli/10_lbl_0_unlbl.json -k transform_type=None seed=0 few_shot_gen_num_lbl=10
python code/normal_train.py -c config/qnli/100_lbl_0_unlbl.json -k transform_type=None seed=0 few_shot_gen_num_lbl=100
python code/normal_train.py -c config/qnli/10_lbl_0_unlbl.json -k transform_type=None seed=1 few_shot_gen_num_lbl=10
python code/normal_train.py -c config/qnli/100_lbl_0_unlbl.json -k transform_type=None seed=1 few_shot_gen_num_lbl=100
python code/normal_train.py -c config/qnli/10_lbl_0_unlbl.json -k transform_type=None seed=42 few_shot_gen_num_lbl=10
python code/normal_train.py -c config/qnli/100_lbl_0_unlbl.json -k transform_type=None seed=42 few_shot_gen_num_lbl=100

python code/normal_train.py -c config/mnli/10_lbl_0_unlbl.json -k transform_type=None seed=0 few_shot_gen_num_lbl=10
python code/normal_train.py -c config/mnli/100_lbl_0_unlbl.json -k transform_type=None seed=0 few_shot_gen_num_lbl=100
python code/normal_train.py -c config/mnli/10_lbl_0_unlbl.json -k transform_type=None seed=1 few_shot_gen_num_lbl=10
python code/normal_train.py -c config/mnli/100_lbl_0_unlbl.json -k transform_type=None seed=1 few_shot_gen_num_lbl=100
python code/normal_train.py -c config/mnli/10_lbl_0_unlbl.json -k transform_type=None seed=42 few_shot_gen_num_lbl=10
python code/normal_train.py -c config/mnli/100_lbl_0_unlbl.json -k transform_type=None seed=42 few_shot_gen_num_lbl=100

python code/normal_train.py -c config/qqp/10_lbl_0_unlbl.json -k transform_type=None seed=0 few_shot_gen_num_lbl=10
python code/normal_train.py -c config/qqp/100_lbl_0_unlbl.json -k transform_type=None seed=0 few_shot_gen_num_lbl=100
python code/normal_train.py -c config/qqp/10_lbl_0_unlbl.json -k transform_type=None seed=1 few_shot_gen_num_lbl=10
python code/normal_train.py -c config/qqp/100_lbl_0_unlbl.json -k transform_type=None seed=1 few_shot_gen_num_lbl=100
python code/normal_train.py -c config/qqp/10_lbl_0_unlbl.json -k transform_type=None seed=42 few_shot_gen_num_lbl=10
python code/normal_train.py -c config/qqp/100_lbl_0_unlbl.json -k transform_type=None seed=42 few_shot_gen_num_lbl=100

python code/normal_train.py -c config/pubmed/10_lbl_0_unlbl.json -k transform_type=None seed=0 few_shot_gen_num_lbl=10
python code/normal_train.py -c config/pubmed/100_lbl_0_unlbl.json -k transform_type=None seed=0 few_shot_gen_num_lbl=100
python code/normal_train.py -c config/pubmed/10_lbl_0_unlbl.json -k transform_type=None seed=1 few_shot_gen_num_lbl=10
python code/normal_train.py -c config/pubmed/100_lbl_0_unlbl.json -k transform_type=None seed=1 few_shot_gen_num_lbl=100
python code/normal_train.py -c config/pubmed/10_lbl_0_unlbl.json -k transform_type=None seed=42 few_shot_gen_num_lbl=10
python code/normal_train.py -c config/pubmed/100_lbl_0_unlbl.json -k transform_type=None seed=42 few_shot_gen_num_lbl=100