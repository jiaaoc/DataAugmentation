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


def pickle_bias():
    labeled_csv = os.path.join("datasets", "bias_data", "WNC", "biased.word.train")

    train_input = []
    with open(labeled_csv, 'r') as f:
        f.readline()
        for line in f.readlines():
            tab_split = line.strip('\n').split('\t')
            train_input.append(tab_split[1])
            train_input.append(tab_split[2])

    train_pk_file = os.path.join("processed_data", "bias_train_unlabeled_data.pkl")
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

def generate_pickle(dataset):
    if dataset == "20_ng":
        pickle_20_ng()
    elif dataset == "hs":
        pickle_hs()
    elif dataset == "bias":
        pickle_bias()
    elif dataset == "pubmed":
        pickle_pubmed()


def generate_csv_hs():
    labeled_csv = os.path.join("datasets", "hatespeech", "labeled_data.csv")

    train_pk_file = os.path.join("processed_data", "hs_train_unlabeled_data.pkl")
    with open(train_pk_file, 'rb') as f:
        original_data = pickle.load(f)

    # Align index wtih index for augmented data
    input = []
    ctr = 0
    with open(labeled_csv, 'r') as f:
        f.readline()
        for line in f.readlines():
            comma_split = line.strip('\n').split(',')
            if len(comma_split) > 5:
                if len(comma_split[0]) > 0:
                    text = ','.join(comma_split[6:])
                    idx = int(comma_split[0])
                    lbl = str(int(comma_split[5])+1)
                    input.append([lbl, str(ctr), text])
            ctr += 1

    num_train = int(0.6 * len(input))

    train_csv = os.path.join("processed_data", "hs", "train.csv")
    with open(train_csv, 'w+') as f:
        for lbl_idx_text in input[:num_train]:
            f.write(','.join(lbl_idx_text) + '\n')

    test_csv = os.path.join("processed_data", "hs", "test.csv")
    with open(test_csv, 'w+') as f:
        for lbl_idx_text in input[num_train:]:
            f.write(','.join(lbl_idx_text) + '\n')

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

def generate_bias():

    labeled_csv = os.path.join("datasets", "bias_data", "WNC", "biased.word.train")

    train_txt = []
    train_lbl = []

    with open(labeled_csv, 'r') as f:
        f.readline()
        for line in f.readlines():
            tab_split = line.strip('\n').split('\t')
            train_txt.append(tab_split[1])
            train_lbl.append(1)
            train_txt.append(tab_split[2])
            train_lbl.append(0)

    ctr = 0
    train_input = []
    for text, lbl in zip(train_txt, train_lbl):
        text = text.replace('\n', '')
        train_input.append([str(lbl+1), str(ctr), text])
        ctr += 1

    train_csv = os.path.join("processed_data", "bias", "train.csv")
    with open(train_csv, 'w+') as f:
        for lbl_idx_text in train_input:
            f.write(','.join(lbl_idx_text) + '\n')

    labeled_csv = os.path.join("datasets", "bias_data", "WNC", "biased.word.dev")

    dev_txt = []
    dev_lbl = []

    with open(labeled_csv, 'r') as f:
        f.readline()
        for line in f.readlines():
            tab_split = line.strip('\n').split('\t')
            dev_txt.append(tab_split[1])
            dev_lbl.append(1)
            dev_txt.append(tab_split[2])
            dev_lbl.append(0)

    dev_input = []
    for text, lbl in zip(train_txt, train_lbl):
        text = text.replace('\n', '')
        dev_input.append([str(lbl+1), str(ctr), text])
        ctr += 1

    dev_csv = os.path.join("processed_data", "bias", "dev.csv")
    with open(dev_csv, 'w+') as f:
        for lbl_idx_text in dev_input:
            f.write(','.join(lbl_idx_text) + '\n')


    labeled_csv = os.path.join("datasets", "bias_data", "WNC", "biased.word.test")

    test_txt = []
    test_lbl = []

    with open(labeled_csv, 'r') as f:
        f.readline()
        for line in f.readlines():
            tab_split = line.strip('\n').split('\t')
            test_txt.append(tab_split[1])
            test_lbl.append(1)
            test_txt.append(tab_split[2])
            test_lbl.append(0)

    test_input = []
    for text, lbl in zip(train_txt, train_lbl):
        text = text.replace('\n', '')
        test_input.append([str(lbl+1), str(ctr), text])
        ctr += 1

    test_csv = os.path.join("processed_data", "bias", "test.csv")
    with open(test_csv, 'w+') as f:
        for lbl_idx_text in dev_input:
            f.write(','.join(lbl_idx_text) + '\n')


def generate_csv(dataset):
    if dataset == "hs":
        generate_csv_hs()
    elif dataset == "20_ng":
        generate_csv_20_ng()
    elif dataset == "bias":
        generate_bias()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--dataset", required=True, choices=['20_ng', 'hs', 'bias', 'pubmed'])
    args = parser.parse_args()

    # generate_pickle(args.dataset)
    generate_csv(args.dataset)
