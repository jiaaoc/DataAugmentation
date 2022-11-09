import os
import numpy as np
import argparse
import pickle
import csv
import codecs

from code.read_data import get_data
from sklearn.datasets import fetch_20newsgroups
from code.Config import Config


def pickle_mrpc(num_lbl):
    config = Config(f"config/mrpc/{num_lbl}_lbl_0_unlbl.json", {})
    train_labeled_set, _, _, _, _ = get_data(config)

    train_pk_file = os.path.join("processed_data", "MRPC",  f"train_{num_lbl}_labeled_data.pkl")
    dict_train_input = {}
    ctr = 0

    for input, output in zip(train_labeled_set.text, train_labeled_set.labels):
        dict_train_input[ctr] = (input, output)
        ctr += 1
    with open(train_pk_file, 'wb') as handle:
        pickle.dump(dict_train_input, handle, protocol=pickle.HIGHEST_PROTOCOL)


def generate_pickle(dataset):
    # if dataset == "20_ng":
    #     pickle_20_ng()
    # elif dataset == "pubmed":
    #     pickle_pubmed()
    # elif dataset == "mnli":
    #     pickle_mnli()
    # elif dataset == "rte":
    #     pickle_rte()
    # elif dataset == "qnli":
    #     pickle_qnli()
    # elif dataset == "qqp":
    #     pickle_qqp()
    if dataset == "mrpc":
        pickle_mrpc(10)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--dataset", required=True)
    args = parser.parse_args()

    generate_pickle(args.dataset)
