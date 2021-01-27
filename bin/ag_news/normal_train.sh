export DA_ROOT=`pwd`
export PYTHONPATH=$DA_ROOT:$PYTHONPATH
export PYTHON_EXEC=python




for i in 0 1 42
do
    CUDA_VISIBLE_DEVICES=6 python -m code.normal_train -c ./config/ag_news/10_lbl_0_unlbl.json -k transform_type=None seed=$i

    CUDA_VISIBLE_DEVICES=6 python -m code.normal_train -c ./config/ag_news/10_lbl_0_unlbl.json -k transform_type=SynonymReplacement seed=$i

    CUDA_VISIBLE_DEVICES=6 python -m code.normal_train -c ./config/ag_news/10_lbl_0_unlbl.json -k transform_type=WordReplacementVocab seed=$i

    CUDA_VISIBLE_DEVICES=6 python -m code.normal_train -c ./config/ag_news/10_lbl_0_unlbl.json -k transform_type=RandomInsertion seed=$i

    CUDA_VISIBLE_DEVICES=6 python -m code.normal_train -c ./config/ag_news/10_lbl_0_unlbl.json -k transform_type=RandomDeletion seed=$i

    CUDA_VISIBLE_DEVICES=6 python -m code.normal_train -c ./config/ag_news/10_lbl_0_unlbl.json -k transform_type=RandomSwapping seed=$i

    CUDA_VISIBLE_DEVICES=6 python -m code.normal_train -c ./config/ag_news/10_lbl_0_unlbl.json -k transform_type=BackTranslation seed=$i

    CUDA_VISIBLE_DEVICES=6 python -m code.normal_train -c ./config/ag_news/10_lbl_0_unlbl.json -k transform_type=Cutoff seed=$i

done




for i in 0 1 42
do
    CUDA_VISIBLE_DEVICES=6 python -m code.normal_train -c ./config/ag_news/100_lbl_0_unlbl.json -k transform_type=None seed=$i

    CUDA_VISIBLE_DEVICES=6 python -m code.normal_train -c ./config/ag_news/100_lbl_0_unlbl.json -k transform_type=SynonymReplacement seed=$i

    CUDA_VISIBLE_DEVICES=6 python -m code.normal_train -c ./config/ag_news/100_lbl_0_unlbl.json -k transform_type=WordReplacementVocab seed=$i

    CUDA_VISIBLE_DEVICES=6 python -m code.normal_train -c ./config/ag_news/100_lbl_0_unlbl.json -k transform_type=RandomInsertion seed=$i

    CUDA_VISIBLE_DEVICES=6 python -m code.normal_train -c ./config/ag_news/100_lbl_0_unlbl.json -k transform_type=RandomDeletion seed=$i

    CUDA_VISIBLE_DEVICES=6 python -m code.normal_train -c ./config/ag_news/100_lbl_0_unlbl.json -k transform_type=RandomSwapping seed=$i

    CUDA_VISIBLE_DEVICES=6 python -m code.normal_train -c ./config/ag_news/100_lbl_0_unlbl.json -k transform_type=BackTranslation seed=$i

    CUDA_VISIBLE_DEVICES=6 python -m code.normal_train -c ./config/ag_news/100_lbl_0_unlbl.json -k transform_type=Cutoff seed=$i

done








