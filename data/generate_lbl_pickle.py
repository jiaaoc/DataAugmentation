import os
import numpy as np
import argparse
import pickle
import csv
import codecs

from code.read_data import get_data
from code.Config import Config


def pickle_dataset(config_dataset, processed_dataset, num_lbl):
    config = Config(f"config/{config_dataset}/{num_lbl}_lbl_0_unlbl.json", {})
    train_labeled_set, _, _, _, _ = get_data(config)

    train_pk_file = os.path.join("processed_data", f"{processed_dataset}",  f"train_{num_lbl}_labeled_data.pkl")
    dict_train_input = {}
    ctr = 0

    for input, output in zip(train_labeled_set.text, train_labeled_set.labels):
        dict_train_input[ctr] = (input, output)
        ctr += 1
    with open(train_pk_file, 'wb') as handle:
        pickle.dump(dict_train_input, handle, protocol=pickle.HIGHEST_PROTOCOL)


def generate_pickle():

    pickle_dataset("20_ng", "20_ng", 10)
    pickle_dataset("20_ng", "20_ng", 100)

    pickle_dataset("pubmed", "pubmed", 10)
    pickle_dataset("pubmed", "pubmed", 100)

    pickle_dataset("mnli", "MNLI", 10)
    pickle_dataset("mnli", "MNLI", 100)

    pickle_dataset("qnli", "QNLI", 10)
    pickle_dataset("qnli", "QNLI", 100)

    pickle_dataset("qqp", "QQP", 10)
    pickle_dataset("qqp", "QQP", 100)

    pickle_dataset("mrpc", "MRPC", 10)
    pickle_dataset("mrpc", "MRPC", 100)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--dataset")
    args = parser.parse_args()

    generate_pickle()
