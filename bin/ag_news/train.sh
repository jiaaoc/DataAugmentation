export DA_ROOT=`pwd`
export PYTHONPATH=$DA_ROOT:$PYTHONPATH
export PYTHON_EXEC=python

CUDA_VISIBLE_DEVICES=7 python -m code.train -c ./config/ag_news/10_lbl_5000_unlbl.json -k transform_type=BackTranslation seed=42