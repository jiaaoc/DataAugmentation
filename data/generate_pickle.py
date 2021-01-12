import os
import numpy as np
import argparse
import pickle

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

    train_dir = os.path.join("processed_data", "imdb", "train")
    pos_dir = os.path.join(train_dir, "pos")
    neg_dir = os.path.join(train_dir, "neg")
    unsup_dir = os.path.join(train_dir, "unsup")

    pos_input = read_dataset(pos_dir)
    print("Finished pos")
    neg_input = read_dataset(neg_dir)
    print("Finished neg")
    unsup_input = read_dataset(unsup_dir)
    print("Finishe unsup")


    train_pk_file = os.path.join("processed_data", "train_unlabeled_data.pkl")
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

def generate_pickle(dataset):
    if dataset == "imdb":
        pickle_imdb()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--dataset", required=True, choices=['imdb'])
    args = parser.parse_args()

    generate_pickle(args.dataset)
