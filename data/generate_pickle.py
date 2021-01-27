import os
import numpy as np
import argparse
import pickle
import csv
import codecs


from sklearn.datasets import fetch_20newsgroups

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

def pickle_pubmed():
    labeled_csv = os.path.join("datasets", "pubmed", "PubMed_20k_RCT_numbers_replaced_with_at_sign", "train.txt")

    train_input = []
    with open(labeled_csv, 'r') as f:
        for line in f.readlines():
            tab_split = line.strip('\n').split('\t')
            if len(tab_split) > 1:
                train_input.append(tab_split[1])

    print(len(train_input))

    train_pk_file = os.path.join("processed_data", "pubmed_train_unlabeled_data.pkl")
    dict_train_input = {}

    ctr = 0
    for text in train_input:
        dict_train_input[ctr] = text
        ctr += 1

    with open(train_pk_file, 'wb') as handle:
        pickle.dump(dict_train_input, handle, protocol=pickle.HIGHEST_PROTOCOL)

def pickle_mnli():
    def read_tsv(filepath):
        list_txt = []

        with open(filepath, 'r') as f:
            # Read header path
            f.readline()

            for idx, line in enumerate(f.readlines()):
                tab_split = line.strip('\n').split('\t')

                sentence_1 = tab_split[8]
                sentence_2 = tab_split[9]

                list_txt.append([sentence_1, sentence_2])

        return list_txt

    train_txt = read_tsv(os.path.join("processed_data", "MNLI", "train.tsv"))

    train_pk_file = os.path.join("processed_data", "MNLI",  "train_unlabeled_data.pkl")
    dict_train_input = {}

    ctr = 0
    for text in train_txt:
        dict_train_input[ctr] = text
        ctr += 1

    with open(train_pk_file, 'wb') as handle:
        pickle.dump(dict_train_input, handle, protocol=pickle.HIGHEST_PROTOCOL)

def pickle_rte():

    def read_tsv(filepath):
        list_txt = []

        with open(filepath, 'r') as f:
            # Read header path
            f.readline()

            for idx, line in enumerate(f.readlines()):
                tab_split = line.strip('\n').split('\t')

                sentence_1 = tab_split[1]
                sentence_2 = tab_split[2]

                list_txt.append([sentence_1, sentence_2])

        return list_txt

    train_txt = read_tsv(os.path.join("processed_data", "RTE", "train.tsv"))

    train_pk_file = os.path.join("processed_data", "RTE",  "train_unlabeled_data.pkl")
    dict_train_input = {}

    ctr = 0
    for text in train_txt:
        dict_train_input[ctr] = text
        ctr += 1

    with open(train_pk_file, 'wb') as handle:
        pickle.dump(dict_train_input, handle, protocol=pickle.HIGHEST_PROTOCOL)




def generate_pickle(dataset):
    if dataset == "20_ng":
        pickle_20_ng()
    elif dataset == "pubmed":
        pickle_pubmed()
    elif dataset == "mnli":
        pickle_mnli()
    elif dataset == "rte":
        pickle_rte()



def generate_csv_20_ng():
    newsgroups_train = fetch_20newsgroups(subset='train')
    train_txt = newsgroups_train.data
    train_lbl = newsgroups_train.target

    ctr = 0
    train_input = []
    for text, lbl in zip(train_txt, train_lbl):
        text = text.replace('\n', '')
        train_input.append([str(lbl+1), str(ctr), text])
        ctr += 1

    train_csv = os.path.join("processed_data", "20_ng", "train.csv")
    with open(train_csv, 'w+') as f:
        for lbl_idx_text in train_input:
            f.write(','.join(lbl_idx_text) + '\n')

    newsgroups_test = fetch_20newsgroups(subset='test')
    test_txt = newsgroups_test.data
    test_lbl = newsgroups_test.target

    test_input = []
    for text, lbl in zip(test_txt, test_lbl):
        text = text.replace('\n', '')
        test_input.append([str(lbl+1), str(ctr), text])
        ctr += 1

    test_csv = os.path.join("processed_data", "20_ng", "test.csv")
    with open(test_csv, 'w+') as f:
        for lbl_idx_text in test_input:
            f.write(','.join(lbl_idx_text) + '\n')



def generate_pubmed():
    train_csv = os.path.join("datasets", "pubmed", "PubMed_20k_RCT_numbers_replaced_with_at_sign", "train.txt")

    dict_lbl2idx = {}
    ctr = 0

    train_input = []
    with open(train_csv, 'r') as f:
        for line in f.readlines():
            tab_split = line.strip('\n').split('\t')
            if len(tab_split) > 1:
                text = tab_split[1]

                if tab_split[0] not in dict_lbl2idx:
                    dict_lbl2idx[tab_split[0]] = len(dict_lbl2idx)
                lbl = dict_lbl2idx[tab_split[0]]
                train_input.append([str(lbl), str(ctr), text])
                ctr += 1

    train_csv = os.path.join("processed_data", "pubmed", "train.csv")
    with open(train_csv, 'w+') as f:
        for lbl_idx_text in train_input:
            f.write(','.join(lbl_idx_text) + '\n')

    dev_csv = os.path.join("datasets", "pubmed", "PubMed_20k_RCT_numbers_replaced_with_at_sign", "dev.txt")
    dev_input = []
    with open(dev_csv, 'r') as f:
        for line in f.readlines():
            tab_split = line.strip('\n').split('\t')
            if len(tab_split) > 1:
                text = tab_split[1]

                if tab_split[0] not in dict_lbl2idx:
                    dict_lbl2idx[tab_split[0]] = len(dict_lbl2idx)
                lbl = dict_lbl2idx[tab_split[0]]
                dev_input.append([str(lbl), str(ctr), text])
                ctr += 1

    dev_csv = os.path.join("processed_data", "pubmed", "dev.csv")
    with open(dev_csv, 'w+') as f:
        for lbl_idx_text in dev_input:
            f.write(','.join(lbl_idx_text) + '\n')

    test_csv = os.path.join("datasets", "pubmed", "PubMed_20k_RCT_numbers_replaced_with_at_sign", "dev.txt")
    test_input = []
    with open(test_csv, 'r') as f:
        for line in f.readlines():
            tab_split = line.strip('\n').split('\t')
            if len(tab_split) > 1:
                text = tab_split[1]

                if tab_split[0] not in dict_lbl2idx:
                    dict_lbl2idx[tab_split[0]] = len(dict_lbl2idx)
                lbl = dict_lbl2idx[tab_split[0]]
                test_input.append([str(lbl), str(ctr), text])
                ctr += 1
    print(dict_lbl2idx)
    test_csv = os.path.join("processed_data", "pubmed", "test.csv")
    with open(test_csv, 'w+') as f:
        for lbl_idx_text in test_input:
            f.write(','.join(lbl_idx_text) + '\n')


def generate_csv(dataset):
    if dataset == "20_ng":
        generate_csv_20_ng()
    elif dataset == "pubmed":
        generate_pubmed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--dataset", required=True)
    args = parser.parse_args()

    generate_pickle(args.dataset)
    # generate_csv(args.dataset)
