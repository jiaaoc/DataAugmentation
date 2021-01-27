import argparse
import os
import json
import numpy as np
from scipy import stats

def get_dict_test_scores(exp_dir):

    dict_seed_to_test_acc = {}
    dict_seed_to_test_f1 = {}

    for seed_dir_name in os.listdir(exp_dir):
        seed = int(seed_dir_name.replace("seed_", ""))
        test_score_file = os.path.join(exp_dir, seed_dir_name, "test_scores_%d.json" % seed)

        with open(test_score_file, 'r') as f:
            first_line = f.readline()
            score_json = json.loads(first_line)
            test_acc = score_json["best_test_acc"]
            test_f1 = score_json["best_test_f1"]

            dict_seed_to_test_acc[seed] = test_acc
            dict_seed_to_test_f1[seed] = test_f1

    average = np.mean(np.asarray(list(dict_seed_to_test_acc.values())))
    std_error = stats.sem((np.asarray(list(dict_seed_to_test_acc.values()))))
    n = len(list(dict_seed_to_test_f1.values()))
    ci = std_error * stats.t.ppf((1 + 0.95) / 2., n - 1)

    print("Average: %.3f, Std Error: %.3f, CI: %.3f " % (average, std_error, ci))
    print(dict_seed_to_test_acc)

    average = np.mean(np.asarray(list(dict_seed_to_test_f1.values())))
    std_dev = scipy.stats.sem(np.asarray(list(dict_seed_to_test_f1.values())))
    ci = std_error * scipy.stats.t.ppf((1 + 0.95) / 2., n - 1)

    print("Average: %.3f, Std Dev: %.3f, CI: %.3f " % (average, std_dev, ci))
    print(dict_seed_to_test_f1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Data Augmentation')
    parser.add_argument('-e', '--exp_dir', required=True)
    args = parser.parse_args()

    get_dict_test_scores(args.exp_dir)
