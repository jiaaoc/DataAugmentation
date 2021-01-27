import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import Dataset

from transformers import *
# from pytorch_transformers import *
import torch.utils.data as Data
import pickle
from data.augmentation import synonym_replacement, random_flip, random_insert, random_delete, word_flip, span_cutoff




class Augmentor:
    """Add different Augmentation here: Synonym Replacement, Word Replacement from Vocab, Random Insertion/deletion/swapping, Word Replacement from LM, BackTranslation
    """

    def __init__(self, config, path=None, transform_type='BackTranslation', transform_times = 1):

        self.transform_type = transform_type
        self.transform_times = transform_times

        self.set_wrds = set()

        if transform_type == 'SynonymReplacement':
            pass
        elif transform_type == 'WordReplacementVocab':
            if "ag_news" in config.dataset.lower():
                train_txt, _, _, _, _ = get_ag_news_data(config)
            elif "20_ng" in config.dataset.lower():
                train_txt, _, _, _ = get_twenty_ng_data(config)
            elif "yahoo" in config.dataset.lower():
                train_txt, _, _, _, _ = get_yahoo_data(config)
            elif "pubmed" in config.dataset.lower():
                train_txt, _, _, _ = get_pubmed_data(config)
            elif "mnli" in config.dataset.lower():
                train_txt, _, _, _ = get_mnli_data(config)
            elif "qqp" in config.dataset.lower():
                train_txt, _, _, _ = get_qqp_data(config)
            elif "sst-2" in config.dataset.lower():
                train_txt, _, _, _ = get_sst2_data(config)
            elif "mrpc" in config.dataset.lower():
                train_txt, _, _, _ = get_mprc_data()
            elif "stsb" in config.dataset.lower():
                train_txt, _, _, _ = get_stsb_data()
            elif "qnli" in config.dataset.lower():
                train_txt, _, _, _ = get_qnli_data(config)
            elif "rte" in config.dataset.lower():
                train_txt, _, _, _ = get_rte_data(config)
            elif "cola" in config.dataset.lower():
                train_txt, _, _, _ = get_cola_data()
            else:
                raise ValueError("Invalid Dataset Name %s" % config.dataset)

            # Here we only use the bodies and removed titles to do the classifications
            for list_txt in train_txt:
                for txt in list_txt:
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
            if 'ag_news' in path:
                with open(path + '/ag_news_de_labeled.pkl', 'rb') as f:
                    de = pickle.load(f)
                    
                with open(path + '/ag_news_de_unlabeled.pkl', 'rb') as f:
                    de_u = pickle.load(f)
                
                de.update(de_u)
                self.transform.append(de)
            else:
                with open(path + 'de_1.pkl', 'rb') as f:
                    de = pickle.load(f)
                    self.transform.append(de)

    def __call__(self, ori, ori_2=None, idx=0):
        augmented_data = []
        augmented_data_2 = None
        if self.transform_type == 'SynonymReplacement':
            augmented_data = synonym_replacement(ori, 0.1, self.transform_times)
            if ori_2 is not None:
                augmented_data_2 = synonym_replacement(ori_2, 0.1, self.transform_times)
        elif self.transform_type == 'WordReplacementVocab':
            augmented_data = word_flip(ori, 0.1, self.transform_times, self.set_wrds)
            if ori_2 is not None:
                augmented_data_2 = word_flip(ori_2, 0.1, self.transform_times, self.set_wrds)
        elif self.transform_type == 'RandomInsertion':
            augmented_data = random_insert(ori, 0.1, self.transform_times)
            if ori_2 is not None:
                augmented_data_2 = random_insert(ori_2, 0.1, self.transform_times)
        elif self.transform_type == 'RandomDeletion':
            augmented_data = random_delete(ori, 0.1, self.transform_times)
            if ori_2 is not None:
                augmented_data_2 = random_delete(ori_2, 0.1, self.transform_times)
        elif self.transform_type == 'RandomSwapping':
            augmented_data = random_flip(ori, 0.1, self.transform_times)
            if ori_2 is not None:
                augmented_data_2 = random_flip(ori_2, 0.1, self.transform_times)
        elif self.transform_type == 'WordReplacementLM':
            augmented_data_2 = []
            if ori_2 is None:
                for i in range(0, self.transform_times):
                    augmented_data.append(self.transform[i][idx])
                while len(augmented_data[0]) < 2:
                    augmented_data[0].append(['.'])
            else:
                for i in range(0, self.transform_times):
                    augmented_data.append(self.transform[i][idx][0][0])
                for i in range(0, self.transform_times):
                    augmented_data_2.append(self.transform[i][idx][1][0])

        elif self.transform_type == 'BackTranslation':
            augmented_data_2 = []
            if ori_2 is None:
                for i in range(0, self.transform_times):
                    augmented_data.append(self.transform[i][idx])
            else:
                for i in range(0, self.transform_times):
                    augmented_data.append(self.transform[i][idx][0][0])
                for i in range(0, self.transform_times):
                    augmented_data_2.append(self.transform[i][idx][1][0])
        elif self.transform_type == "Cutoff":
            augmented_data = span_cutoff(ori, 0.1, self.transform_times)
            if ori_2 is not None:
                augmented_data_2 = span_cutoff(ori_2, 0.1, self.transform_times)

        return augmented_data, augmented_data_2, ori

def get_twenty_ng_data(config):
    def read_csv(filepath):
        list_lbl = []
        list_txt = []

        with open(filepath, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                comma_split = line.strip('\n').split(',')
                if len(comma_split) > 2 and comma_split[0].isdigit() and comma_split[1].isdigit():
                    list_lbl.append(int(comma_split[0]) - 1)
                    list_txt.append([','.join(comma_split[2:])])

        return list_txt, list_lbl

    train_txt, train_lbl = read_csv(os.path.join(config.datapath, "train.csv"))
    test_txt, test_lbl = read_csv(os.path.join(config.datapath, "test.csv"))

    return np.asarray(train_txt), np.asarray(train_lbl), np.asarray(test_txt), np.asarray(test_lbl)

def get_qnli_data(config):
    dict_lbl_2_idx = {"not_entailment": 0, "entailment": 1}

    def read_tsv(filepath):
        list_lbl = []
        list_txt = []

        with open(filepath, 'r') as f:
            # Read header path
            f.readline()

            for idx, line in enumerate(f.readlines()):
                tab_split = line.strip('\n').split('\t')

                question = tab_split[1]
                sentence = tab_split[2]
                lbl = int(dict_lbl_2_idx[tab_split[3]])

                list_txt.append([question, sentence])
                list_lbl.append(lbl)

        return list_txt, list_lbl

    train_txt, train_lbl = read_tsv(os.path.join(config.datapath, "train.tsv"))
    test_txt, test_lbl = read_tsv(os.path.join(config.datapath, "dev.tsv"))

    return np.asarray(train_txt), np.asarray(train_lbl), np.asarray(test_txt), np.asarray(test_lbl)


def get_mnli_data(config):
    dict_lbl_2_idx = {"contradiction":0, "neutral": 1, "entailment": 2}

    def read_tsv(filepath):
        list_lbl = []
        list_txt = []

        with open(filepath, 'r') as f:
            # Read header path
            f.readline()

            for idx, line in enumerate(f.readlines()):
                tab_split = line.strip('\n').split('\t')

                sentence_1 = tab_split[8]
                sentence_2 = tab_split[9]
                lbl = int(dict_lbl_2_idx[tab_split[11]])

                list_txt.append([sentence_1, sentence_2])
                list_lbl.append(lbl)

        return list_txt, list_lbl

    train_txt, train_lbl = read_tsv(os.path.join(config.datapath, "train.tsv"))
    test_txt, test_lbl = read_tsv(os.path.join(config.datapath, "dev_matched.tsv"))

    return np.asarray(train_txt), np.asarray(train_lbl), np.asarray(test_txt), np.asarray(test_lbl)



def get_ag_news_data(config):
    
    train_df = pd.read_csv(os.path.join(config.datapath, "train.csv"), header=None)
    test_df = pd.read_csv(os.path.join(config.datapath, "test.csv"), header=None)


    if config.n_labeled_per_class == -1:
        train_labels = np.array([v-1 for v in train_df[0]])
        train_text = np.array([[v] for v in train_df[2]])
        del train_df

        test_labels = np.array([u-1 for u in test_df[0]])
        test_text = np.array([[v] for v in test_df[2]])
        del test_df

        return train_text, train_labels, test_text, test_labels, None

    else:
        train_labels = np.array([v-1 for v in train_df[0]])
        train_text = np.array([[v] for v in train_df[2]])
        del train_df

        test_labels = np.array([u-1 for u in test_df[0]])
        test_text = np.array([[v] for v in test_df[2]])
        del test_df

        n_labels = max(test_labels) + 1
        np.random.seed(0)

        train_idx_pool = []

        for i in range(n_labels):
            idxs = np.where(train_labels == i)[0]
            np.random.shuffle(idxs)
            train_idx_pool.extend(idxs[:1000 + 20000])

        return train_text, train_labels, test_text, test_labels, train_idx_pool



def get_yahoo_data(config):
    
    train_df = pd.read_csv(os.path.join(config.datapath, "train.csv"), header=None)
    test_df = pd.read_csv(os.path.join(config.datapath, "test.csv"), header=None)


    if config.n_labeled_per_class == -1:
        train_labels = np.array([v-1 for v in train_df[0]])
        train_text = np.array([[v] for v in train_df[2]])
        del train_df

        test_labels = np.array([u-1 for u in test_df[0]])
        test_text = np.array([[v] for v in test_df[2]])
        del test_df

        return train_text, train_labels, test_text, test_labels, None

    else:
        train_labels = np.array([v-1 for v in train_df[0]])
        train_text = np.array([[v] for v in train_df[2]])
        del train_df

        test_labels = np.array([u-1 for u in test_df[0]])
        test_text = np.array([[v] for v in test_df[2]])
        del test_df

        n_labels = max(test_labels) + 1
        np.random.seed(0)

        train_idx_pool = []

        for i in range(n_labels):
            idxs = np.where(train_labels == i)[0]
            np.random.shuffle(idxs)
            train_idx_pool.extend(idxs[:1000 + 20000])

        return train_text, train_labels, test_text, test_labels, train_idx_pool

def get_qqp_data(config):
    def read_tsv(filepath):
        list_lbl = []
        list_txt = []

        with open(filepath, 'r') as f:
            # Read header path
            f.readline()

            for idx, line in enumerate(f.readlines()):
                tab_split = line.strip('\n').split('\t')

                question_1 = tab_split[3]
                question_2 = tab_split[4]
                lbl = int(tab_split[5])

                list_txt.append([question_1, question_2])
                list_lbl.append(lbl)

        return list_txt, list_lbl

    train_txt, train_lbl = read_tsv(os.path.join(config.datapath, "train.tsv"))
    test_txt, test_lbl = read_tsv(os.path.join(config.datapath, "dev.tsv"))

    return np.asarray(train_txt), np.asarray(train_lbl), np.asarray(test_txt), np.asarray(test_lbl)

def get_rte_data(config):
    dict_lbl_2_idx = {"not_entailment":0, "entailment": 1}

    def read_tsv(filepath):
        list_lbl = []
        list_txt = []

        with open(filepath, 'r') as f:
            # Read header path
            f.readline()

            for idx, line in enumerate(f.readlines()):
                tab_split = line.strip('\n').split('\t')

                sentence_1 = tab_split[1]
                sentence_2 = tab_split[2]
                lbl = int(dict_lbl_2_idx[tab_split[3]])

                list_txt.append([sentence_1, sentence_2])
                list_lbl.append(lbl)

        return list_txt, list_lbl

    train_txt, train_lbl = read_tsv(os.path.join(config.datapath, "train.tsv"))
    test_txt, test_lbl = read_tsv(os.path.join(config.datapath, "dev.tsv"))

    return np.asarray(train_txt), np.asarray(train_lbl), np.asarray(test_txt), np.asarray(test_lbl)


def get_pubmed_data(config):
    def read_csv(filepath):
        list_lbl = []
        list_txt = []

        with open(filepath, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                comma_split = line.strip('\n').split(',')
                if len(comma_split) > 2 and comma_split[0].isdigit() and comma_split[1].isdigit():
                    list_lbl.append(int(comma_split[0]) - 1)
                    list_txt.append([','.join(comma_split[2:])])

        return list_txt, list_lbl

    train_txt, train_lbl = read_csv(os.path.join(config.datapath, "train.csv"))
    test_txt, test_lbl = read_csv(os.path.join(config.datapath, "test.csv"))

    return np.asarray(train_txt)[:130000], np.asarray(train_lbl)[:130000], np.asarray(test_txt), np.asarray(test_lbl)


def get_sst2_data(config):
    train_df = pd.read_csv(os.path.join(config.datapath, "train.tsv"), sep = '\t')
    test_df = pd.read_csv(os.path.join(config.datapath, "dev.tsv"), sep = '\t')

    train_labels = np.array([v for v in train_df['label']])
    train_text = np.array([[v] for v in train_df['sentence']])
    
    del train_df
    
    test_labels = np.array([v for v in test_df['label']])
    test_text = np.array([[v] for v in test_df['sentence']])

    del test_df

    return train_text, train_labels, test_text, test_labels


def get_data(config):

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
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    train_idx_pool = None
    # Labels must be 0 indexed
    # All datasets must return lists of lists
    if "ag_news" in config.dataset.lower():
        train_txt, train_labels, test_txt, test_lbl, train_idx_pool = get_ag_news_data(config)
    elif "20_ng" in config.dataset.lower():
        train_txt, train_labels, test_txt, test_lbl = get_twenty_ng_data(config)
    elif "yahoo" in config.dataset.lower():
        train_txt, train_labels, test_txt, test_lbl, train_idx_pool = get_yahoo_data(config)
    elif "pubmed" in config.dataset.lower():
        train_txt, train_labels, test_txt, test_lbl = get_pubmed_data(config)
    elif "mnli" in config.dataset.lower():
        train_txt, train_labels, test_txt, test_lbl = get_mnli_data(config)
    elif "qqp" in config.dataset.lower():
        train_txt, train_labels, test_txt, test_lbl = get_qqp_data(config)
    elif "sst-2" in config.dataset.lower():
        train_txt, train_labels, test_txt, test_lbl = get_sst2_data(config)
    elif "mrpc" in config.dataset.lower():
        train_txt, train_labels, test_txt, test_lbl = get_mprc_data()
    elif "stsb" in config.dataset.lower():
        train_txt, train_labels, test_txt, test_lbl = get_stsb_data()
    elif "qnli" in config.dataset.lower():
        train_txt, train_labels, test_txt, test_lbl = get_qnli_data(config)
    elif "rte" in config.dataset.lower():
        train_txt, train_labels, test_txt, test_lbl = get_rte_data(config)
    elif "cola" in config.dataset.lower():
        train_txt, train_labels, test_txt, test_lbl = get_cola_data()
    else:
        raise ValueError("Invalid Dataset Name %s" % config.dataset)

    n_labels = max(test_lbl) + 1

    train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(
         train_labels, config.n_labeled_per_class, config.unlabeled_per_class, n_labels, config.datapath, config.seed, train_idx_pool)


    augmentor = None
    if config.transform_type is not None:
        augmentor = Augmentor(config, config.datapath, config.transform_type, config.transform_times)

    train_labeled_dataset = loader_labeled(
    train_txt[train_labeled_idxs], train_labels[train_labeled_idxs], train_labeled_idxs, tokenizer, config.max_seq_length, augmentor)

    train_unlabeled_dataset = loader_unlabeled(
    train_txt[train_unlabeled_idxs], train_unlabeled_idxs, tokenizer, config.max_seq_length, Augmentor(config, config.datapath, config.transform_type, config.transform_times))

    val_dataset = loader_labeled(train_txt[val_idxs], train_labels[val_idxs], val_idxs, tokenizer, config.max_seq_length)

    test_dataset = loader_labeled(test_txt, test_lbl, None, tokenizer, config.max_seq_length)

    print("#Labeled: {}, Unlabeled {}, Val {}, Test {}".format(len(
    train_labeled_idxs), len(train_unlabeled_idxs), len(val_idxs), len(test_lbl)))

    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset, n_labels


def train_val_split(labels, n_labeled_per_class, unlabeled_per_class, n_labels, dataset_path, seed=0, train_idx_pool = None):
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


    data_seed_path = os.path.join(dataset_path, "seed_%d_lbl_%d_unlbl_%d" % (seed, n_labeled_per_class, unlabeled_per_class))

    if os.path.exists(data_seed_path):
        train_labeled_idxs = pickle.load(open(os.path.join(data_seed_path, "train_lbl_idx.pkl"), 'rb'))
        train_unlabeled_idxs = pickle.load(open(os.path.join(data_seed_path, "train_unlbl_idx.pkl"), 'rb'))
        val_idxs = pickle.load(open(os.path.join(data_seed_path, "val_idx.pkl"), 'rb'))
    else:
        if train_idx_pool is not None:
            labels = np.array(labels)
            
            train_labeled_idxs = []
            train_unlabeled_idxs = []

            num_data = len(train_idx_pool)

            num_val = min(int(0.2 * num_data), 2000)

            np.random.seed(seed)
            
            
            rand_perm = np.array(train_idx_pool.copy())
            np.random.shuffle(rand_perm)

            val_idxs = rand_perm[-num_val:]

            rand_perm_labels = labels[rand_perm]

            for i in range(len(rand_perm_labels) - num_val):
                temp_lbl_idxs = np.where(rand_perm_labels == i)[0]
                lbl_idxs = rand_perm[temp_lbl_idxs]
                train_labeled_idxs.extend(lbl_idxs[:n_labeled_per_class])
                train_unlabeled_idxs.extend(lbl_idxs[100:min(100+unlabeled_per_class, len(lbl_idxs))])

            np.random.shuffle(train_labeled_idxs)
            np.random.shuffle(train_unlabeled_idxs)
            np.random.shuffle(val_idxs)

            if not os.path.exists(data_seed_path):
                os.makedirs(data_seed_path)

            pickle.dump(train_labeled_idxs, open(os.path.join(data_seed_path, "train_lbl_idx.pkl"), 'wb+'))
            pickle.dump(train_unlabeled_idxs, open(os.path.join(data_seed_path, "train_unlbl_idx.pkl"), 'wb+'))
            pickle.dump(val_idxs, open(os.path.join(data_seed_path, "val_idx.pkl"), 'wb+'))
        else:
            labels = np.array(labels)
            train_labeled_idxs = []
            train_unlabeled_idxs = []

            num_data = len(labels)

            num_val = min(int(0.2 * num_data), 2000)

            np.random.seed(seed)
            rand_perm = np.arange(num_data)
            np.random.shuffle(rand_perm)

            val_idxs = rand_perm[-num_val:]

            if n_labeled_per_class == -1:
                train_labeled_idxs = rand_perm[:-num_val].tolist()
                train_unlabeled_idxs = [0] # prevent crash
            else:
                rand_perm_labels = labels[rand_perm]

                for i in range(len(rand_perm_labels) - num_val):
                    temp_lbl_idxs = np.where(rand_perm_labels == i)[0]
                    lbl_idxs = rand_perm[temp_lbl_idxs]
                    train_labeled_idxs.extend(lbl_idxs[:n_labeled_per_class])
                    train_unlabeled_idxs.extend(lbl_idxs[100:min(100+unlabeled_per_class, len(lbl_idxs))])

                np.random.shuffle(train_labeled_idxs)
                np.random.shuffle(train_unlabeled_idxs)
                np.random.shuffle(val_idxs)

                if not os.path.exists(data_seed_path):
                    os.makedirs(data_seed_path)

                pickle.dump(train_labeled_idxs, open(os.path.join(data_seed_path, "train_lbl_idx.pkl"), 'wb+'))
                pickle.dump(train_unlabeled_idxs, open(os.path.join(data_seed_path, "train_unlbl_idx.pkl"), 'wb+'))
                pickle.dump(val_idxs, open(os.path.join(data_seed_path, "val_idx.pkl"), 'wb+'))

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs

class loader_labeled(Dataset):
    # Data loader for labeled data
    def __init__(self, dataset_text, dataset_label, dataset_idx, tokenizer, max_seq_len, augmentor = None):
        self.tokenizer = tokenizer
        self.text = dataset_text
        self.labels = dataset_label
        self.ids = dataset_idx
        self.max_seq_len = max_seq_len

        self.augmentor = augmentor

        if augmentor is not None:
            print('Augment training data')
            self.augmentor = augmentor

    def __len__(self):
        return len(self.labels)

    def get_double_tokenized(self, text_1, text_2):
        tokens = self.tokenizer(text_1, text_2, padding="max_length", truncation="longest_first", add_special_tokens=True, max_length=self.max_seq_len)["input_ids"]
        return tokens

    def get_tokenized(self, text):
        tokens = self.tokenizer(text, padding="max_length", truncation=True, add_special_tokens=True, max_length=self.max_seq_len)["input_ids"]
        return tokens

    def __getitem__(self, idx):
        tokenized_data = []
        labels = []

        if self.augmentor is not None:
            ori = self.text[idx]

            if len(ori) == 2:
                ori_a = ori[0]; ori_b = ori[1]

                augmented_data_a, augmented_data_b, _ = self.augmentor(ori_a, ori_b, self.ids[idx])

                encode_result_u = self.get_double_tokenized(augmented_data_a, augmented_data_b)
                tokenized_data.append(torch.tensor(encode_result_u))
                labels.append(self.labels[idx])

                encode_result_ori = self.get_double_tokenized(ori_a, ori_b)
                tokenized_data.append(torch.tensor(encode_result_ori)[None,:])
                labels.append(self.labels[idx])

                labels = torch.tensor(labels)
                tokenized_data = torch.cat(tokenized_data, dim=0) # [bs*2, max_seq_len]

            else:
                ori = ori[0]
                augmented_data, _, _ = self.augmentor(ori, idx=self.ids[idx])
                encode_result_u = self.get_tokenized(augmented_data)
                tokenized_data.append(torch.tensor(encode_result_u))
                labels.append(self.labels[idx])

                encode_result_ori = self.get_tokenized(ori)
                tokenized_data.append(torch.tensor(encode_result_ori)[None,:])
                labels.append(self.labels[idx])

                labels = torch.tensor(labels)

                tokenized_data = torch.cat(tokenized_data, dim=0) # [bs*2, max_seq_len]


        else:
            ori = self.text[idx]

            if len(ori) == 2:
                ori_a = ori[0]; ori_b = ori[1]

                tokenized_data = self.get_double_tokenized(ori_a, ori_b)
                labels.append(self.labels[idx])

                labels = torch.tensor(labels)
                tokenized_data = torch.tensor(tokenized_data) # [bs, max_seq_len]

            else:
                ori = ori[0]
                tokenized_data = self.get_tokenized(ori)
                labels.append(self.labels[idx])
                labels = torch.tensor(labels)
                tokenized_data = torch.tensor(tokenized_data) # [bs, max_seq_len]


        return tokenized_data, labels




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

    def get_double_tokenized(self, text_1, text_2):
        tokens = self.tokenizer(text_1, text_2, padding="max_length", truncation="longest_first", add_special_tokens=True, max_length=self.max_seq_len)["input_ids"]
        return tokens

    def get_tokenized(self, text):
        tokens = self.tokenizer(text, padding="max_length", truncation=True, add_special_tokens=True, max_length=self.max_seq_len)["input_ids"]
        return tokens

    def __getitem__(self, idx):
        ori = self.text[idx]


        if len(ori) == 2:
            ori_a = ori[0]
            ori_b = ori[1]

            augmented_data_a, augmented_data_b, _ = self.augmentor(ori_a, ori_b, self.ids[idx])
            encode_result_u = self.get_double_tokenized(augmented_data_a, augmented_data_b)[0]

            encode_result_ori = self.get_double_tokenized(ori_a, ori_b)

        else:
            ori = ori[0]
            augmented_data, _, _ = self.augmentor(ori, idx=self.ids[idx])

            encode_result_u = self.get_tokenized(augmented_data)[0]
            encode_result_ori = self.get_tokenized(ori)

        return torch.tensor(encode_result_ori), torch.tensor(encode_result_u)
