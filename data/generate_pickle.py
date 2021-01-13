import os
import numpy as np
import argparse
import pickle
import csv

from sklearn.datasets import fetch_20newsgroups

def pickle_imdb():
    def read_dataset(split_dir):
        '''
        Read the dataset

        :param dataset_dir:
        :return: list_text
        '''
        input = []

        for idx, filename in enumerate(os.listdir(split_dir)):
            with open(os.path.join(split_dir, filename), 'r', encoding='utf-8') as f:
                line = f.readline().replace("<br /><br />", " ")
                input.append(line.strip('\n'))
            print("Finished %d " % idx, end='\r')

        return np.asarray(input)

    train_dir = os.path.join("datasets", "imdb", "train")
    pos_dir = os.path.join(train_dir, "pos")
    neg_dir = os.path.join(train_dir, "neg")
    unsup_dir = os.path.join(train_dir, "unsup")

    pos_input = read_dataset(pos_dir)
    neg_input = read_dataset(neg_dir)
    unsup_input = read_dataset(unsup_dir)

    train_pk_file = os.path.join("processed_data", "imdb_train_unlabeled_data.pkl")
    dict_train_input = {}

    ctr = 0
    for text in pos_input[:5000]:
        dict_train_input[ctr] = text
        ctr += 1
    for text in neg_input[:5000]:
        dict_train_input[ctr] = text
        ctr += 1
    for text in unsup_input[:20000]:
        dict_train_input[ctr] = text
        ctr += 1

    with open(train_pk_file, 'wb') as handle:
        pickle.dump(dict_train_input, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_ag_news():
    def read_dataset(split):
        '''
        Read the dataset

        :param dataset_dir:
        :return: list_text
        '''
        split_file = os.path.join("datasets", "ag_news", "%s.csv" % split)

        input = []

        with open(split_file, encoding='utf-8') as f:
            f.readline()

            for line in f.readlines():
                comma_split = line.strip('\n').split(',')
                lbl = comma_split[0]
                title = comma_split[1]
                description = comma_split[2:]

                inp_txt = title + ",".join(description)

                input.append(inp_txt)

        return np.asarray(input)

    train_pk_file = os.path.join("processed_data", "ag_news_train_unlabeled_data.pkl")
    dict_train_input = {}

    ctr = 0
    train_input = read_dataset("train")
    for text in train_input:
        dict_train_input[ctr] = text
        ctr += 1

    with open(train_pk_file, 'wb') as handle:
        pickle.dump(dict_train_input, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_20_ng():
    newsgroups_train = fetch_20newsgroups(subset='train')

    train_input = newsgroups_train.data

    train_pk_file = os.path.join("processed_data", "20_ng_train_unlabeled_data.pkl")
    dict_train_input = {}

    ctr = 0
    for text in train_input:
        dict_train_input[ctr] = text
        ctr += 1

    with open(train_pk_file, 'wb') as handle:
        pickle.dump(dict_train_input, handle, protocol=pickle.HIGHEST_PROTOCOL)

def pickle_hs():
    labeled_csv = os.path.join("datasets", "hatespeech", "labeled_data.csv")

    train_input = []
    with open(labeled_csv, 'r') as f:
        f.readline()

        for line in f.readlines():
            comma_split = line.strip('\n').split(',')
            text = ','.join(comma_split[6:])
            train_input.append(text)

    train_pk_file = os.path.join("processed_data", "hs_train_unlabeled_data.pkl")
    dict_train_input = {}

    ctr = 0
    for text in train_input:
        dict_train_input[ctr] = text
        ctr += 1

    with open(train_pk_file, 'wb') as handle:
        pickle.dump(dict_train_input, handle, protocol=pickle.HIGHEST_PROTOCOL)


def generate_pickle(dataset):
    if dataset == "imdb":
        pickle_imdb()
    elif dataset == "ag_news":
        pickle_ag_news()
    elif dataset == "20_ng":
        pickle_20_ng()
    elif dataset == "hs":
        pickle_hs()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--dataset", required=True, choices=['imdb', 'ag_news', '20_ng', 'hs'])
    args = parser.parse_args()

    generate_pickle(args.dataset)
