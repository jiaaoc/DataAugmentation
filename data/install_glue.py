import os
import json
import argparse

from datasets import load_dataset, load_metric







def install_dataset(dataset_name):
    datasets = load_dataset("glue", dataset_name)

    train_dataset = datasets["train"]

    val_match_dataset = datasets["validation_matched"]
    val_mismatch_dataset = datasets["validation_mismatched"]
    test_match_dataset = datasets["test_matched"]
    test_mismatch_dataset = datasets["test_mismatched"]

    dataset_dir = os.path.join("processed_data", "mnli")

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)


    with open(os.path.join(dataset_dir, "train.csv"), 'w+') as f:
        for idx, point in enumerate(train_dataset):
            f.write(json.dumps(point))
            print("Finished %d" % idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--dataset", required=True)
    args = parser.parse_args()

    install_dataset(args.dataset)
