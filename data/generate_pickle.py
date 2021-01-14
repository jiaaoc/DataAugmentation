import os
import numpy as np
import argparse
import pickle
import csv
import codecs


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
    if dataset == "imdb":
        pickle_imdb()
    elif dataset == "ag_news":
        pickle_ag_news()
    elif dataset == "20_ng":
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
        test_input.append([str(lbl), str(ctr), text])
        ctr += 1

    test_csv = os.path.join("processed_data", "20_ng", "test.csv")
    with open(test_csv, 'w+') as f:
        for lbl_idx_text in test_input:
            f.write(','.join(lbl_idx_text) + '\n')


def generate_csv(dataset):
    if dataset == "hs":
        generate_csv_hs()
    elif dataset == "20_ng":
        generate_csv_20_ng()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--dataset", required=True, choices=['imdb', 'ag_news', '20_ng', 'hs', 'bias', 'pubmed'])
    args = parser.parse_args()

    # generate_pickle(args.dataset)
    generate_csv(args.dataset)
