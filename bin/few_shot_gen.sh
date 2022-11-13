#!/usr/bin/env bash


python data/prompt_pred.py --dataset qnli -d processed_data/QNLI
python data/prompt_pred.py --dataset qnli -d processed_data/QNLI -n 100

python data/prompt_pred.py --dataset mnli -d processed_data/MNLI
python data/prompt_pred.py --dataset mnli -d processed_data/MNLI -n 100

python data/prompt_pred.py --dataset qqp -d processed_data/QQP
python data/prompt_pred.py --dataset qqp -d processed_data/QQP -n 100

python data/prompt_pred.py --dataset 20_ng -d processed_data/20_ng
python data/prompt_pred.py --dataset 20_ng -d processed_data/20_ng -n 100

python data/prompt_pred.py --dataset pubmed -d processed_data/pubmed
python data/prompt_pred.py --dataset pubmed -d processed_data/pubmed -n 100
