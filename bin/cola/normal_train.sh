export DA_ROOT=`pwd`
export PYTHONPATH=$DA_ROOT:$PYTHONPATH
export PYTHON_EXEC=python


#CUDA_VISIBLE_DEVICES=2 python -m code.normal_train -c ./config/cola/10_lbl_0_unlbl.json -k n_labeled_per_class=-1 epochs=10 transform_type=None seed=42


#CUDA_VISIBLE_DEVICES=2 python -m code.normal_train -c ./config/cola/10_lbl_0_unlbl.json -k n_labeled_per_class=-1 lr=1e-5 epochs=10 transform_type=None seed=42


for i in 0 1 42
do
    CUDA_VISIBLE_DEVICES=2 python -m code.normal_train -c ./config/cola/10_lbl_0_unlbl.json -k transform_type=None seed=$i

    CUDA_VISIBLE_DEVICES=2 python -m code.normal_train -c ./config/cola/10_lbl_0_unlbl.json -k transform_type=SynonymReplacement seed=$i

    CUDA_VISIBLE_DEVICES=2 python -m code.normal_train -c ./config/cola/10_lbl_0_unlbl.json -k transform_type=WordReplacementVocab seed=$i

    CUDA_VISIBLE_DEVICES=2 python -m code.normal_train -c ./config/cola/10_lbl_0_unlbl.json -k transform_type=RandomInsertion seed=$i

    CUDA_VISIBLE_DEVICES=2 python -m code.normal_train -c ./config/cola/10_lbl_0_unlbl.json -k transform_type=RandomDeletion seed=$i

    CUDA_VISIBLE_DEVICES=2 python -m code.normal_train -c ./config/cola/10_lbl_0_unlbl.json -k transform_type=RandomSwapping seed=$i

    CUDA_VISIBLE_DEVICES=2 python -m code.normal_train -c ./config/cola/10_lbl_0_unlbl.json -k transform_type=BackTranslation seed=$i

    CUDA_VISIBLE_DEVICES=2 python -m code.normal_train -c ./config/cola/10_lbl_0_unlbl.json -k transform_type=Cutoff seed=$i

done




for i in 0 1 42
do
    CUDA_VISIBLE_DEVICES=2 python -m code.normal_train -c ./config/cola/100_lbl_0_unlbl.json -k transform_type=None seed=$i

    CUDA_VISIBLE_DEVICES=2 python -m code.normal_train -c ./config/cola/100_lbl_0_unlbl.json -k transform_type=SynonymReplacement seed=$i

    CUDA_VISIBLE_DEVICES=2 python -m code.normal_train -c ./config/cola/100_lbl_0_unlbl.json -k transform_type=WordReplacementVocab seed=$i

    CUDA_VISIBLE_DEVICES=2 python -m code.normal_train -c ./config/cola/100_lbl_0_unlbl.json -k transform_type=RandomInsertion seed=$i

    CUDA_VISIBLE_DEVICES=2 python -m code.normal_train -c ./config/cola/100_lbl_0_unlbl.json -k transform_type=RandomDeletion seed=$i

    CUDA_VISIBLE_DEVICES=2 python -m code.normal_train -c ./config/cola/100_lbl_0_unlbl.json -k transform_type=RandomSwapping seed=$i

    CUDA_VISIBLE_DEVICES=2 python -m code.normal_train -c ./config/cola/100_lbl_0_unlbl.json -k transform_type=BackTranslation seed=$i

    CUDA_VISIBLE_DEVICES=2 python -m code.normal_train -c ./config/cola/100_lbl_0_unlbl.json -k transform_type=Cutoff seed=$i

done






for i in 42
do

    CUDA_VISIBLE_DEVICES=2 python -m code.train -c ./config/cola/10_lbl_5000_unlbl.json -k transform_type=SynonymReplacement seed=$i

    CUDA_VISIBLE_DEVICES=2 python -m code.train -c ./config/cola/10_lbl_5000_unlbl.json -k transform_type=WordReplacementVocab seed=$i

    CUDA_VISIBLE_DEVICES=2 python -m code.train -c ./config/cola/10_lbl_5000_unlbl.json -k transform_type=RandomInsertion seed=$i

    CUDA_VISIBLE_DEVICES=2 python -m code.train -c ./config/cola/10_lbl_5000_unlbl.json -k transform_type=RandomDeletion seed=$i

    CUDA_VISIBLE_DEVICES=2 python -m code.train -c ./config/cola/10_lbl_5000_unlbl.json -k transform_type=RandomSwapping seed=$i

    CUDA_VISIBLE_DEVICES=2 python -m code.train -c ./config/cola/10_lbl_5000_unlbl.json -k transform_type=BackTranslation seed=$i

    CUDA_VISIBLE_DEVICES=2 python -m code.train -c ./config/cola/10_lbl_5000_unlbl.json -k transform_type=Cutoff seed=$i

done




for i in 42
do

    CUDA_VISIBLE_DEVICES=2 python -m code.train -c ./config/cola/100_lbl_5000_unlbl.json -k transform_type=SynonymReplacement seed=$i

    CUDA_VISIBLE_DEVICES=2 python -m code.train -c ./config/cola/100_lbl_5000_unlbl.json -k transform_type=WordReplacementVocab seed=$i

    CUDA_VISIBLE_DEVICES=2 python -m code.train -c ./config/cola/100_lbl_5000_unlbl.json -k transform_type=RandomInsertion seed=$i

    CUDA_VISIBLE_DEVICES=2 python -m code.train -c ./config/cola/100_lbl_5000_unlbl.json -k transform_type=RandomDeletion seed=$i

    CUDA_VISIBLE_DEVICES=2 python -m code.train -c ./config/cola/100_lbl_5000_unlbl.json -k transform_type=RandomSwapping seed=$i

    CUDA_VISIBLE_DEVICES=2 python -m code.train -c ./config/cola/100_lbl_5000_unlbl.json -k transform_type=BackTranslation seed=$i

    CUDA_VISIBLE_DEVICES=2 python -m code.train -c ./config/cola/100_lbl_5000_unlbl.json -k transform_type=Cutoff seed=$i

done




for i in 0 1
do

    CUDA_VISIBLE_DEVICES=2 python -m code.train -c ./config/cola/10_lbl_5000_unlbl.json -k transform_type=SynonymReplacement seed=$i

    CUDA_VISIBLE_DEVICES=2 python -m code.train -c ./config/cola/10_lbl_5000_unlbl.json -k transform_type=WordReplacementVocab seed=$i

    CUDA_VISIBLE_DEVICES=2 python -m code.train -c ./config/cola/10_lbl_5000_unlbl.json -k transform_type=RandomInsertion seed=$i

    CUDA_VISIBLE_DEVICES=2 python -m code.train -c ./config/cola/10_lbl_5000_unlbl.json -k transform_type=RandomDeletion seed=$i

    CUDA_VISIBLE_DEVICES=2 python -m code.train -c ./config/cola/10_lbl_5000_unlbl.json -k transform_type=RandomSwapping seed=$i

    CUDA_VISIBLE_DEVICES=2 python -m code.train -c ./config/cola/10_lbl_5000_unlbl.json -k transform_type=BackTranslation seed=$i

    CUDA_VISIBLE_DEVICES=2 python -m code.train -c ./config/cola/10_lbl_5000_unlbl.json -k transform_type=Cutoff seed=$i

done




for i in 0 1
do

    CUDA_VISIBLE_DEVICES=2 python -m code.train -c ./config/cola/100_lbl_5000_unlbl.json -k transform_type=SynonymReplacement seed=$i

    CUDA_VISIBLE_DEVICES=2 python -m code.train -c ./config/cola/100_lbl_5000_unlbl.json -k transform_type=WordReplacementVocab seed=$i

    CUDA_VISIBLE_DEVICES=2 python -m code.train -c ./config/cola/100_lbl_5000_unlbl.json -k transform_type=RandomInsertion seed=$i

    CUDA_VISIBLE_DEVICES=2 python -m code.train -c ./config/cola/100_lbl_5000_unlbl.json -k transform_type=RandomDeletion seed=$i

    CUDA_VISIBLE_DEVICES=2 python -m code.train -c ./config/cola/100_lbl_5000_unlbl.json -k transform_type=RandomSwapping seed=$i

    CUDA_VISIBLE_DEVICES=2 python -m code.train -c ./config/cola/100_lbl_5000_unlbl.json -k transform_type=BackTranslation seed=$i

    CUDA_VISIBLE_DEVICES=2 python -m code.train -c ./config/cola/100_lbl_5000_unlbl.json -k transform_type=Cutoff seed=$i

done






