import argparse
import pickle
import os



def check_backtranslation(original_file, backtranslation_file):
    with open(original_file, 'rb') as f:
        original_data = pickle.load(f)

    with open(backtranslation_file, 'rb') as f:
        backtranslation_data = pickle.load(f)

    ctr = 0
    for (k, v) in backtranslation_data.items():
        if len(v) > 0:
            ctr += 1
    print(ctr)


def update_backtranslation(backtranslation_file):
    with open(backtranslation_file, 'rb') as f:
        backtranslation_data = pickle.load(f)

    new_backtranslation_data = {}
    ctr = 0
    for (k, v) in backtranslation_data.items():
        if len(v) > 0:
            new_backtranslation_data[ctr] = v
            ctr += 1


def rte_backtranslation():

    def read_tsv(filepath):
        dict_txt = {}
        ctr = 0
        with open(filepath, 'r') as f:
            # Read header path
            f.readline()

            for idx, line in enumerate(f.readlines()):
                tab_split = line.strip('\n').split('\t')

                sentence_1 = tab_split[1]
                sentence_2 = tab_split[7]

                dict_txt[ctr] = [sentence_1, sentence_2]
                ctr += 1

        return dict_txt

    dict_txt = read_tsv(os.path.join("glue_bt_data", "rte", "train_bt.tsv"))

    with open("processed_data/RTE/" + 'de_1.pkl', 'wb') as f:
        pickle.dump(dict_txt, f)



def qnli_backtranslation():

    def read_tsv(filepath):
        dict_txt = {}
        ctr = 0
        with open(filepath, 'r') as f:
            # Read header path
            f.readline()

            for idx, line in enumerate(f.readlines()):
                tab_split = line.strip('\n').split('\t')

                sentence_1 = tab_split[1]
                sentence_2 = tab_split[7]

                dict_txt[ctr] = [sentence_1, sentence_2]
                ctr += 1

        return dict_txt

    dict_txt = read_tsv(os.path.join("glue_bt_data", "qnli", "train_bt.tsv"))

    with open("processed_data/QNLI/" + 'de_1.pkl', 'wb') as f:
        pickle.dump(dict_txt, f)



def qqp_backtranslation():

    def read_tsv(filepath):
        dict_txt = {}
        ctr = 0
        with open(filepath, 'r') as f:
            # Read header path
            f.readline()

            for idx, line in enumerate(f.readlines()):
                tab_split = line.strip('\n').split('\t')

                sentence_1 = tab_split[1]
                sentence_2 = tab_split[7]

                dict_txt[ctr] = [sentence_1, sentence_2]
                ctr += 1

        return dict_txt

    dict_txt = read_tsv(os.path.join("glue_bt_data", "qqp", "train_bt.tsv"))

    with open("processed_data/QQP/" + 'de_1.pkl', 'wb') as f:
        pickle.dump(dict_txt, f)

def mnli_backtranslation():

    def read_tsv(filepath):
        dict_txt = {}
        ctr = 0
        with open(filepath, 'r') as f:
            # Read header path
            f.readline()

            for idx, line in enumerate(f.readlines()):
                tab_split = line.strip('\n').split('\t')

                sentence_1 = tab_split[1]
                sentence_2 = tab_split[7]

                dict_txt[ctr] = [sentence_1, sentence_2]
                ctr += 1

        return dict_txt

    dict_txt = read_tsv(os.path.join("glue_bt_data", "mnli", "train_bt.tsv"))

    with open("processed_data/MNLI/" + 'de_1.pkl', 'wb') as f:
        pickle.dump(dict_txt, f)


def glue_backtranslation(dataset):
    if dataset == "rte":
        return rte_backtranslation()
    elif dataset == "qnli":
        return qnli_backtranslation()
    elif dataset == "qqp":
        return qqp_backtranslation()
    elif dataset == "mnli":
        return mnli_backtranslation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', "--original_file")
    parser.add_argument('-b', "--back_translation_file")
    parser.add_argument('-d', "--dataset")
    args = parser.parse_args()

    # check_backtranslation(args.original_file, args.back_translation_file)
    # update_backtranslation(args.back_translation_file)
    glue_backtranslation(args.dataset)