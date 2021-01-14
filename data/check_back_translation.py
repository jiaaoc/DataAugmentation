import argparse
import pickle




def check_backtranslation(original_file, backtranslation_file):
    with open(original_file, 'rb') as f:
        original_data = pickle.load(f)

    with open(backtranslation_file, 'rb') as f:
        backtranslation_data = pickle.load(f)

    for (k, v) in original_data.items():
        print(k, v, backtranslation_data[k])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', "--original_file", required=True)
    parser.add_argument('-b', "--back_translation_file", required=True)
    args = parser.parse_args()

    check_backtranslation(args.original_file, args.back_translation_file)