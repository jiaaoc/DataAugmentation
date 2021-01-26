import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import Dataset
from datasets import load_dataset, load_metric

from transformers import *
# from pytorch_transformers import *
import torch.utils.data as Data
import pickle
from data.augmentation import synonym_replacement, random_flip, random_insert, random_delete, word_flip, span_cutoff



task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}



class GlueAugmentor:
    """Add different Augmentation here: Synonym Replacement, Word Replacement from Vocab, Random Insertion/deletion/swapping, Word Replacement from LM, BackTranslation
    """

    def __init__(self, path=None, transform_type='BackTranslation', transform_times=1):

        self.transform_type = transform_type
        self.transform_times = transform_times

        self.set_wrds = set()

        if transform_type == 'SynonymReplacement':
            pass
        elif transform_type == 'WordReplacementVocab':
            if "hs" in path or "bias" in path or "20_ng" in path or "pubmed" in path:
                train_df = read_csv(path + 'train.csv')
            else:
                train_df = pd.read_csv(path + 'train.csv', header=None)

            # Here we only use the bodies and removed titles to do the classifications
            for txt in train_df[2]:
                self.set_wrds.update(txt.split(' '))

            self.set_wrds = list(self.set_wrds)

        elif transform_type == 'RandomInsertion':
            pass
        elif transform_type == 'RandomDeletion':
            pass
        elif transform_type == 'RandomSwapping':
            pass
        elif transform_type == 'WordReplacementLM':
            self.transform = []
            # Pre-processed German data
            with open(path + 'mlm.pkl', 'rb') as f:
                mlm = pickle.load(f)
                self.transform.append(mlm)

        elif transform_type == 'BackTranslation':
            self.transform = []
            # Pre-processed German data
            with open(path + 'de_1.pkl', 'rb') as f:
                de = pickle.load(f)
                self.transform.append(de)
            # # Pre-processed Russian data
            # with open(path + 'ru_1.pkl', 'rb') as f:
            #     ru = pickle.load(f)
            #     self.transform.append(ru)

    def __call__(self, ori_1, ori_2, idx):
        augmented_data = []
        augmented_data_2 = []

        if self.transform_type == 'SynonymReplacement':
            augmented_data = synonym_replacement(ori_1, 0.3, self.transform_times)
            augmented_data_2 = synonym_replacement(ori_2, 0.3, self.transform_times)
        elif self.transform_type == 'WordReplacementVocab':
            augmented_data = word_flip(ori_1, 0.3, self.transform_times, self.set_wrds)
            augmented_data_2 = word_flip(ori_2, 0.3, self.transform_times, self.set_wrds)
        elif self.transform_type == 'RandomInsertion':
            augmented_data = random_insert(ori_1, 0.3, self.transform_times)
            augmented_data_2 = random_insert(ori_2, 0.3, self.transform_times)
        elif self.transform_type == 'RandomDeletion':
            augmented_data = random_delete(ori_1, 0.3, self.transform_times)
            augmented_data_2 = random_delete(ori_2, 0.3, self.transform_times)
        elif self.transform_type == 'RandomSwapping':
            augmented_data = random_flip(ori_1, 0.3, self.transform_times)
            augmented_data_2 = random_flip(ori_2, 0.3, self.transform_times)
        elif self.transform_type == 'WordReplacementLM':
            for i in range(0, self.transform_times):
                augmented_data.append(self.transform[0][idx][i])
        elif self.transform_type == 'BackTranslation':
            for i in range(0, self.transform_times):
                augmented_data.append(self.transform[0][idx][i])
        elif self.transform_type == "Cutoff":
            augmented_data = span_cutoff(ori_1, 0.3, self.transform_times)
            augmented_data_2 = span_cutoff(ori_2, 0.3, self.transform_times)


        return augmented_data, augmented_data_2, ori_1, ori_2



class Augmentor:
    """Add different Augmentation here: Synonym Replacement, Word Replacement from Vocab, Random Insertion/deletion/swapping, Word Replacement from LM, BackTranslation
    """

    def __init__(self, path=None, transform_type='BackTranslation', transform_times = 1):

        self.transform_type = transform_type
        self.transform_times = transform_times

        self.set_wrds = set()

        if transform_type == 'SynonymReplacement':
            pass
        elif transform_type == 'WordReplacementVocab':
            if "hs" in path or "bias" in path or "20_ng" in path or "pubmed" in path:
                train_df = read_csv(path + 'train.csv')
            else:
                train_df = pd.read_csv(path + 'train.csv', header=None)

            # Here we only use the bodies and removed titles to do the classifications
            for txt in train_df[2]:
                self.set_wrds.update(txt.split(' '))

            self.set_wrds = list(self.set_wrds)

        elif transform_type == 'RandomInsertion':
            pass
        elif transform_type == 'RandomDeletion':
            pass
        elif transform_type == 'RandomSwapping':
            pass
        elif transform_type == 'WordReplacementLM':
            self.transform = []
            # Pre-processed German data
            with open(path + 'mlm.pkl', 'rb') as f:
                mlm = pickle.load(f)
                self.transform.append(mlm)

        elif transform_type == 'BackTranslation':
            self.transform = []
            # Pre-processed German data
            with open(path + 'de_1.pkl', 'rb') as f:
                de = pickle.load(f)
                self.transform.append(de)
            # # Pre-processed Russian data
            # with open(path + 'ru_1.pkl', 'rb') as f:
            #     ru = pickle.load(f)
            #     self.transform.append(ru)

    def __call__(self, ori, idx):
        augmented_data = []
        
        if self.transform_type == 'SynonymReplacement':
            augmented_data = synonym_replacement(ori, 0.3, self.transform_times)
        elif self.transform_type == 'WordReplacementVocab':
            augmented_data = word_flip(ori, 0.3, self.transform_times, self.set_wrds)
        elif self.transform_type == 'RandomInsertion':
            augmented_data = random_insert(ori, 0.3, self.transform_times)
        elif self.transform_type == 'RandomDeletion':
            augmented_data = random_delete(ori, 0.3, self.transform_times)
        elif self.transform_type == 'RandomSwapping':
            augmented_data = random_flip(ori, 0.3, self.transform_times)
        elif self.transform_type == 'WordReplacementLM':
            for i in range(0, self.transform_times):
                augmented_data.append(self.transform[0][idx][i])
        elif self.transform_type == 'BackTranslation':
            for i in range(0, self.transform_times):
                augmented_data.append(self.transform[0][idx][i])
        elif self.transform_type == "Cutoff":
            augmented_data = span_cutoff(ori, 0.3, self.transform_times)

        return augmented_data, ori


def read_csv(datapath):
     list_labels = []
     list_idx = []
     list_text = []

     with open(datapath, 'r') as f:

         for idx, line in enumerate(f.readlines()):
             comma_split = line.strip('\n').split(',')
             if len(comma_split) > 2 and comma_split[0].isdigit() and comma_split[1].isdigit():
                 list_labels.append(int(comma_split[0]))
                 list_idx.append(int(comma_split[1]))
                 list_text.append(','.join(comma_split[2:]))

     return [list_labels, list_idx, list_text]


def read_glue(datasets, top_k=None):

    if top_k is None:
        datasets_list = datasets[:len(datasets)]
    else:
        datasets_list = datasets[:top_k]



    list_labels = datasets_list["label"]
    list_text_1 = datasets_list["premise"]
    list_text_2 = datasets_list["hypothesis"]

    return [np.asarray(list_labels), np.asarray(list_text_1), np.asarray(list_text_2)]


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
     # print(data_path+'train.csv')

     val_df = None
     if "hs" in data_path or "20_ng" in data_path or "bias" in data_path or "pubmed" in data_path:
         train_df = read_csv(data_path+'train.csv')

         if "pubmed" in data_path:
             train_df = train_df[:130000]

         test_df = read_csv(data_path+'test.csv')
         if os.path.exists(data_path+'dev.csv'):
             val_df = read_csv(data_path+'dev.csv')
             val_labels = np.array([u-1 for u in val_df[0]])
             val_text = np.array([v for v in val_df[2]])

     else:
         train_df = pd.read_csv(data_path+'train.csv', header=None)
         test_df = pd.read_csv(data_path+'test.csv', header=None)

     # Here we only use the bodies and removed titles to do the classifications
     train_labels = np.array([v-1 for v in train_df[0]])
     train_text = np.array([v for v in train_df[2]])

     test_labels = np.array([u-1 for u in test_df[0]])
     test_text = np.array([v for v in test_df[2]])

     n_labels = max(test_labels) + 1

     if val_df is None:
         # Split the labeled training set, unlabeled training set, development set
         train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(
         train_labels, n_labeled_per_class, unlabeled_per_class, n_labels)
     else:
         train_labeled_idxs, train_unlabeled_idxs = train_split(train_labels, n_labeled_per_class, unlabeled_per_class, n_labels)
         val_idxs = np.arange(min(len(val_text), 2000))

     train_labeled_dataset = loader_labeled(
     train_text[train_labeled_idxs], train_labels[train_labeled_idxs], train_labeled_idxs, tokenizer, max_seq_len, train_aug, Augmentor(data_path, transform_type, transform_times))

     train_unlabeled_dataset = loader_unlabeled(
     train_text[train_unlabeled_idxs], train_unlabeled_idxs, tokenizer, max_seq_len, Augmentor(data_path, transform_type, transform_times))
     if val_df is None:
         val_dataset = loader_labeled(
         train_text[val_idxs], train_labels[val_idxs], val_idxs, tokenizer, max_seq_len)
     else:
         val_dataset = loader_labeled(
         val_text[val_idxs], val_labels[val_idxs], val_idxs, tokenizer, max_seq_len)

     test_dataset = loader_labeled(
     test_text, test_labels, None, tokenizer, max_seq_len)

     print("#Labeled: {}, Unlabeled {}, Val {}, Test {}".format(len(
     train_labeled_idxs), len(train_unlabeled_idxs), len(val_idxs), len(test_labels)))

     return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset, n_labels


def get_glue_data(datapath, n_labeled_per_class, unlabeled_per_class=5000, max_seq_len=256, model='bert-base-uncased', train_aug=False, transform_type='BackTranslation', transform_times = 2):

    if "mnli" in datapath:
        datasets = load_dataset("glue", datapath)

    padding = "max_length"

    sentence1_key, sentence2_key = task_to_keys["mnli"]

    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased"
    )

    label_to_id = None

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (
                examples[sentence1_key], examples[sentence2_key])
        )

        result = tokenizer(*args, padding=padding, max_length=256, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [label_to_id[l] for l in examples["label"]]
        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=True)


    test_labels = map(lambda x: x["label"], datasets["test_matched"])
    second_val_labels = None
    second_test_labels = None

    if "mnli" in datapath:

        if n_labeled_per_class == -1:
            top_k = None
        else:
            top_k = (n_labeled_per_class + 20000) * 3
        train_labels, train_text_1, train_text_2 = read_glue(datasets["train"], top_k=top_k)
        val_labels, val_text_1, val_text_2 = read_glue(datasets["validation_matched"])
        second_val_labels, second_val_text_1, second_val_text_2 = read_glue(datasets["validation_mismatched"])
        test_labels, test_text_1, test_text_2 = read_glue(datasets["test_matched"])
        second_test_labels, second_test_text_1, second_test_text_2 = read_glue(datasets["test_mismatched"])

    n_labels = len(set(train_labels))

    train_labeled_idxs, train_unlabeled_idxs = train_split(train_labels, n_labeled_per_class, unlabeled_per_class, n_labels)


    train_labeled_dataset = glue_loader_labeled(
        train_text_1[train_labeled_idxs],  train_text_2[train_labeled_idxs], train_labels[train_labeled_idxs], train_labeled_idxs, tokenizer,
        max_seq_len, train_aug, GlueAugmentor(datapath, transform_type, transform_times))
    #
    # train_unlabeled_dataset = glue_loader_labeled(
    #     train_text_1[train_unlabeled_idxs], train_text_2[train_unlabeled_idxs], tokenizer, max_seq_len,
    #     Augmentor(datapath, transform_type, transform_times))


    num_train = train_labels.shape[0]
    num_val = val_labels.shape[0]
    num_test = test_labels.shape[0]

    val_idx = range(num_train, num_train+num_val)
    test_idx = range(num_train+num_val, num_train+num_val+num_test)

    val_dataset = glue_loader_labeled(
        val_text_1, val_text_2, val_labels, val_idx, tokenizer, max_seq_len)
    second_val_dataset = None
    if second_val_labels is not None:
        second_val_dataset = glue_loader_labeled(
            second_val_text_1, second_val_text_2, second_val_labels, val_idx, tokenizer, max_seq_len)

    test_dataset = glue_loader_labeled(
        test_text_1, test_text_2, test_labels, test_idx, tokenizer, max_seq_len)
    second_test_dataset = None
    if second_test_labels is not None:
        second_test_dataset =  glue_loader_labeled(
        second_test_text_1, second_test_text_2, second_test_labels, test_idx, tokenizer, max_seq_len)

    return train_labeled_dataset, None, val_dataset, second_val_dataset, test_dataset, second_test_dataset, n_labels


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

    num_data = len(labels)

    num_val = int(0.25 * num_data)
    idxs = np.arange(num_data)
    val_idxs = idxs[-num_val:]

    if n_labeled_per_class == -1:
        train_labeled_idxs = idxs[:-num_val]
        train_unlabeled_idxs = []
    else:
        train_labels = labels[:-num_val]
        for i in range(n_labels):
            lbl_idxs = np.where(train_labels == i)[0]
            train_labeled_idxs.extend(lbl_idxs[:n_labeled_per_class])
            train_unlabeled_idxs.extend(lbl_idxs[n_labeled_per_class:n_labeled_per_class+unlabeled_per_class])

    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs


def train_split(labels, n_labeled_per_class, unlabeled_per_class, n_labels, seed=0):
    np.random.seed(seed)
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []

    num_data = len(labels)

    idxs = np.arange(num_data)

    if n_labeled_per_class == -1:
        train_labeled_idxs = idxs
        train_unlabeled_idxs = []
    else:
        train_labels = labels
        for i in range(n_labels):
            lbl_idxs = np.where(train_labels == i)[0]
            train_labeled_idxs.extend(lbl_idxs[:n_labeled_per_class])
            train_unlabeled_idxs.extend(lbl_idxs[n_labeled_per_class:n_labeled_per_class+unlabeled_per_class])

    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)

    return train_labeled_idxs, train_unlabeled_idxs


class loader_labeled(Dataset):
    # Data loader for labeled data
    def __init__(self, dataset_text, dataset_label, dataset_idx, tokenizer, max_seq_len, aug=False, augmentor = None):
        self.tokenizer = tokenizer
        self.text = dataset_text
        self.labels = dataset_label
        self.ids = dataset_idx
        self.max_seq_len = max_seq_len

        self.aug = aug
        self.trans_dist = {}
        self.augmentor = augmentor

        if aug:
            print('Augment training data')
            self.augmentor = augmentor

    def __len__(self):
        return len(self.labels)

    def get_tokenized(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len - 2:
            tokens = tokens[:self.max_seq_len - 2]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
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

            labels = torch.tensor(labels)
            tokenized_data = torch.stack(tokenized_data, dim=0)

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







class glue_loader_labeled(Dataset):
    # Data loader for labeled data
    def __init__(self, dataset_text, dataset_text_2, dataset_label, dataset_idx, tokenizer, max_seq_len, aug=False, augmentor=None):
        self.tokenizer = tokenizer
        self.text = dataset_text
        self.text_2 = dataset_text_2
        self.labels = dataset_label
        self.ids = dataset_idx
        self.max_seq_len = max_seq_len

        self.aug = aug
        self.trans_dist = {}
        self.augmentor = augmentor

        if aug:
            print('Augment training data')
            self.augmentor = augmentor

    def __len__(self):
        return len(self.labels)

    def get_tokenized(self, text_1, text_2):
        tokens = self.tokenizer.tokenize(text_1, text_2, add_special_tokens=True)

        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        length = len(tokens)

        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(encode_result))
        encode_result += padding



        return encode_result, length

    def __getitem__(self, idx):
        if self.augmentor is not None:
            augmented_data, augmented_data_2,  ori, ori_2 = self.augmentor(self.text[idx], self.text_2[idx], self.ids[idx])

            tokenized_data = []
            labels = []
            tokenized_sentence_length = []
            for s1, s2 in zip(augmented_data, augmented_data_2):
                encode_result_u, length_u = self.get_tokenized(s1, s2)
                tokenized_data.append(torch.tensor(encode_result_u))
                labels.append(self.labels[idx])
                tokenized_sentence_length.append(length_u)

            encode_result_ori, length_ori = self.get_tokenized(ori, ori_2)
            tokenized_data.append(torch.tensor(encode_result_ori))
            labels.append(self.labels[idx])
            tokenized_sentence_length.append(length_ori)

            labels = torch.tensor(labels)
            tokenized_data = torch.stack(tokenized_data, dim=0)

            return (tokenized_data, labels, tokenized_sentence_length)

        else:
            text = self.text[idx]
            text_2 = self.text_2[idx]

            tokens = self.tokenizer.tokenize(text, text_2)
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
        if len(tokens) > self.max_seq_len - 2:
            tokens = tokens[:self.max_seq_len - 2]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
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
            # for u in augmented_data:
            #     encode_result_u, length_u = self.get_tokenized(u)
            #     tokenized_data.append(torch.tensor(encode_result_u))
            #     tokenized_sentence_length.append(length_u)
            encode_result_u1, length_u1 = self.get_tokenized(augmented_data[0])
            # tokenized_data.append(torch.tensor(encode_result_u))
            # tokenized_sentence_length.append(length_u)

            encode_result_u2, length_u2 = self.get_tokenized(augmented_data[0])

            encode_result_ori, length_ori = self.get_tokenized(ori)
            # tokenized_data.append(torch.tensor(encode_result_ori))
            # tokenized_sentence_length.append(length_ori)

            # return (tokenized_data, tokenized_sentence_length)
            return (torch.tensor(encode_result_u1), torch.tensor(encode_result_u2), torch.tensor(encode_result_ori)), (length_u1, length_u2, length_ori)

        else:
            text = self.text[idx]
            encode_result, length = self.get_tokenized(text)
            return (torch.tensor(encode_result), length)
