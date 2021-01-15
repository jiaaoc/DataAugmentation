import argparse
import pickle




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
            ctr += 1
            new_backtranslation_data[ctr] = v

    with open('processed_data/hs/de_1.pkl', 'wb+') as f:
        pickle.dump(new_backtranslation_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', "--original_file")
    parser.add_argument('-b', "--back_translation_file", required=True)
    args = parser.parse_args()

    # check_backtranslation(args.original_file, args.back_translation_file)
    update_backtranslation(args.back_translation_file)