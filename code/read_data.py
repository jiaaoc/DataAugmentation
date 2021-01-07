import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pytorch_transformers import *
import torch.utils.data as Data
import pickle


class Augmentor:
    """Add different Augmentation here: Synonym Replacement, Word Replacement from Vocab, Random Insertion/deletion/swapping, Word Replacement from LM, BackTranslation
    """

    def __init__(self, path=None, transform_type='BackTranslation', transform_times = 2):

        self.transform_type = transform_type
        self.transform_times = transform_times

        if transform_type == 'SynonymReplacement':
            pass
        elif transform_type == 'WordReplacementVocab':
            pass
        elif transform_type == 'RandomInsertion':
            pass
        elif transform_type == 'RandomDeletion':
            pass
        elif transform_type == 'RandomSwapping':
            pass
        elif transform_type == 'WordReplacementLM':
            pass
        elif transform_type == 'BackTranslation':
            self.transform = []
            # Pre-processed German data
            with open(path + 'de_1.pkl', 'rb') as f:
                de = pickle.load(f)
                self.transform.append(de)
            # Pre-processed Russian data
            with open(path + 'ru_1.pkl', 'rb') as f:
                ru = pickle.load(f)
                self.transform.append(ru)

    

    def __call__(self, ori, idx):
        augmented_data = []
        
        if self.transform_type == 'SynonymReplacement':
            pass
        elif self.transform_type == 'WordReplacementVocab':
            pass
        elif self.transform_type == 'RandomInsertion':
            pass
        elif self.transform_type == 'RandomDeletion':
            pass
        elif self.transform_type == 'RandomSwapping':
            pass
        elif self.transform_type == 'WordReplacementLM':
            pass
        elif self.transform_type == 'BackTranslation':
            for i in range(0, self.transform_times):
                augmented_data.append(self.transform[i][idx])
            
        return augmented_data, ori

def get_data(data_path, n_labeled_per_class, unlabeled_per_class=5000, max_seq_len=256, model='bert-base-uncased', train_aug=False, transform_type='BackTranslation', transform_times = 2):
    """Read data, split the dataset, and build dataset for dataloaders.

    Arguments:
        data_path {str} -- Path to your dataset folder: contain a train.csv and test.csv
        n_labeled_per_class {int} -- Number of labeled data per class

    Keyword Arguments:
        unlabeled_per_class {int} -- Number of unlabeled data per class (default: {5000})
        max_seq_len {int} -- Maximum sequence length (default: {256})
        model {str} -- Model name (default: {'bert-base-uncased'})
        train_aug {bool} -- Whether performing augmentation on labeled training set (default: {False})

    """
    # Load the tokenizer for bert
    tokenizer = BertTokenizer.from_pretrained(model)

    train_df = pd.read_csv(data_path+'train.csv', header=None)
    test_df = pd.read_csv(data_path+'test.csv', header=None)

    # Here we only use the bodies and removed titles to do the classifications
    train_labels = np.array([v-1 for v in train_df[0]])
    train_text = np.array([v for v in train_df[2]])

    test_labels = np.array([u-1 for u in test_df[0]])
    test_text = np.array([v for v in test_df[2]])

    n_labels = max(test_labels) + 1

    # Split the labeled training set, unlabeled training set, development set
    train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(
        train_labels, n_labeled_per_class, unlabeled_per_class, n_labels)

    # Build the dataset class for each set
    train_labeled_dataset = loader_labeled(
        train_text[train_labeled_idxs], train_labels[train_labeled_idxs], tokenizer, max_seq_len, train_aug, Augmentor(data_path, transform_type, transform_times))
    train_unlabeled_dataset = loader_unlabeled(
        train_text[train_unlabeled_idxs], train_unlabeled_idxs, tokenizer, max_seq_len, Augmentor(data_path, transform_type, transform_times))
    val_dataset = loader_labeled(
        train_text[val_idxs], train_labels[val_idxs], tokenizer, max_seq_len)
    test_dataset = loader_labeled(
        test_text, test_labels, tokenizer, max_seq_len)

    print("#Labeled: {}, Unlabeled {}, Val {}, Test {}".format(len(
        train_labeled_idxs), len(train_unlabeled_idxs), len(val_idxs), len(test_labels)))

    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset, n_labels


def train_val_split(labels, n_labeled_per_class, unlabeled_per_class, n_labels, seed=0):
    """Split the original training set into labeled training set, unlabeled training set, development set

    Arguments:
        labels {list} -- List of labeles for original training set
        n_labeled_per_class {int} -- Number of labeled data per class
        unlabeled_per_class {int} -- Number of unlabeled data per class
        n_labels {int} -- The number of classes

    Keyword Arguments:
        seed {int} -- [random seed of np.shuffle] (default: {0})

    Returns:
        [list] -- idx for labeled training set, unlabeled training set, development set
    """
    np.random.seed(seed)
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(n_labels):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        if n_labels == 2:
            # IMDB
            train_pool = np.concatenate((idxs[:500], idxs[5500:-2000]))
            train_labeled_idxs.extend(train_pool[:n_labeled_per_class])
            train_unlabeled_idxs.extend(
                idxs[500: 500 + 5000])
            val_idxs.extend(idxs[-2000:])
        elif n_labels == 10:
            # DBPedia
            train_pool = np.concatenate((idxs[:500], idxs[10500:-2000]))
            train_labeled_idxs.extend(train_pool[:n_labeled_per_class])
            train_unlabeled_idxs.extend(
                idxs[500: 500 + unlabeled_per_class])
            val_idxs.extend(idxs[-2000:])
        else:
            # Yahoo/AG News
            train_pool = np.concatenate((idxs[:500], idxs[5500:-2000]))
            train_labeled_idxs.extend(train_pool[:n_labeled_per_class])
            train_unlabeled_idxs.extend(
                idxs[500: 500 + 5000])
            val_idxs.extend(idxs[-2000:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs


class loader_labeled(Dataset):
    # Data loader for labeled data
    def __init__(self, dataset_text, dataset_label, tokenizer, max_seq_len, aug=False, augmentor = None):
        self.tokenizer = tokenizer
        self.text = dataset_text
        self.labels = dataset_label
        self.max_seq_len = max_seq_len

        self.aug = aug
        self.trans_dist = {}

        if aug:
            print('Augment training data')
            self.augmentor = augmentor

    def __len__(self):
        return len(self.labels)

    def get_tokenized(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        length = len(tokens)

        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(encode_result))
        encode_result += padding

        return encode_result, length

    def __getitem__(self, idx):
        if self.augmentor is not None:


            augmented_data, ori = self.augmentor(self.text[idx], self.ids[idx])

            tokenized_data = []
            labels = []
            tokenized_sentence_length = []
            for u in augmented_data:
                encode_result_u, length_u = self.get_tokenized(u)
                tokenized_data.append(torch.tensor(encode_result_u))
                labels.append(self.labels[idx])
                tokenized_sentence_length.append(length_u)

            encode_result_ori, length_ori = self.get_tokenized(ori)
            tokenized_data.append(torch.tensor(encode_result_ori))
            labels.append(self.labels[idx])
            tokenized_sentence_length.append(length_ori)

            return (tokenized_data, labels, tokenized_sentence_length)
            
        else:
            text = self.text[idx]
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len]
            length = len(tokens)
            encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
            padding = [0] * (self.max_seq_len - len(encode_result))
            encode_result += padding
            return (torch.tensor(encode_result), self.labels[idx], length)


class loader_unlabeled(Dataset):
    # Data loader for unlabeled data
    def __init__(self, dataset_text, unlabeled_idxs, tokenizer, max_seq_len, augmentor=None):
        self.tokenizer = tokenizer
        self.text = dataset_text
        self.ids = unlabeled_idxs
        self.augmentor = augmentor
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.text)

    def get_tokenized(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        length = len(tokens)
        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(encode_result))
        encode_result += padding
        return encode_result, length

    def __getitem__(self, idx):
        if self.augmentor is not None:
            augmented_data, ori = self.augmentor(self.text[idx], self.ids[idx])

            tokenized_data = []
            tokenized_sentence_length = []
            for u in augmented_data:
                encode_result_u, length_u = self.get_tokenized(u)
                tokenized_data.append(torch.tensor(encode_result_u))
                tokenized_sentence_length.append(length_u)

            encode_result_ori, length_ori = self.get_tokenized(ori)
            tokenized_data.append(torch.tensor(encode_result_ori))
            tokenized_sentence_length.append(length_ori)

            return (tokenized_data, tokenized_sentence_length)
        else:
            text = self.text[idx]
            encode_result, length = self.get_tokenized(text)
            return (torch.tensor(encode_result), length)
