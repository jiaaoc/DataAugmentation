import argparse
import os
import json
import numpy as np

def get_dict_test_scores(exp_dir):

    dict_seed_to_test_acc = {}

    for seed_dir_name in os.listdir(exp_dir):
        seed = int(seed_dir_name.replace("seed_", ""))
        test_score_file = os.path.join(exp_dir, seed_dir_name, "test_scres_%d.json" % seed)

        with open(test_score_file, 'r') as f:
            first_line = f.readline()
            score_json = json.loads(first_line)
            test_acc = score_json["best_test_acc"]

            dict_seed_to_test_acc[seed] = test_acc


    average = np.mean(dict_seed_to_test_acc.values())
    std_dev = np.std(dict_seed_to_test_acc.values())

    print("Average: %.3f, Std Dev: %.3f " % (average, std_dev))
    print(dict_seed_to_test_acc)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Data Augmentation')
    parser.add_argument('-e', '--exp_dir', required=True)
    args = parser.parse_args()