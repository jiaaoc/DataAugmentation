# An Empirical Survey of Data Augmentation for Limited Data Learning in NLP


This repo contains codes for the following paper: 

*Jiaao Chen, Derek Tam, Colin Raffel, Mohit Bansal, Diyi Yang*: An Empirical Survey of Data Augmentation for Limited Data Learning in NLP

If you would like to refer to it, please cite the paper mentioned above. 


### Code Structure
```
|__ data/
        |__ back_translation.py --> Script for back translating the dataset
        |__ augmentation.py --> Script for generating augmentations
        |__ generate_pickle.py --> Pickle data 
        |__ mlm_pred.py --> Augmenting data using MLM 


|__code/
        |__ read_data.py --> Codes for reading the dataset; forming labeled training set, unlabeled training set, development set and testing set; building dataloaders
        |__ normal_train.py --> Codes for training supervised models 
        |__ train.py --> Codes for training SSL models 
        |__ CLS_model.py --> baseline BERT model 
        |__ Config.py --> contains values used for training 
```

### Downloading the data
Please download the dataset and put them in the data folder. You can find processed 20_NG data [here](https://drive.google.com/file/d/1hBbtkLRrHo5b_5APEKSDUD9az6YJ2yhM/view?usp=sharing).

### Pre-processing the data

We first pickle the data. See `generate_pickle.py` for examples. For augmentation methods requiring external LMs, we first generate and cache those augmentations. See `back_translation.py` and `generate_pickle.py` for examples. All other augmentations are done on the fly. 


### Training models
Examples for how to train supervised models can be found in `bin/20_ng/train.sh` and to train semi-supervised models can be found in `bin/20_ng/ssl_train.sh`

