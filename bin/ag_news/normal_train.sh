export DA_ROOT=`pwd`
export PYTHONPATH=$DA_ROOT:$PYTHONPATH
export PYTHON_EXEC=python

CUDA_VISIBLE_DEVICES=7 python -m code.normal_train -c ./config/ag_news/10_lbl_0_unlbl.json -k transform_type=None seed=42