
from code.read_data import *
from code.Config import Config

from sklearn.datasets import fetch_20newsgroups


def main():

    config = Config(os.path.join("config/20_ng/10_lbl_0_unlbl.json"), {"seed": 0})

    train_txt, train_labels, test_txt, test_lbl = get_twenty_ng_data(config)

    n_labels = max(test_lbl) + 1

    train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(
         train_labels, config.n_labeled_per_class, config.unlabeled_per_class, n_labels, config.datapath, config.seed, None)

    train_lbl_txt = train_txt[train_labeled_idxs]
    train_lbl_lbl = train_labels[train_labeled_idxs]

    newsgroups_train = fetch_20newsgroups(subset='train')
    dict_lbl_idx_2_lbl = newsgroups_train.target_names

    train_lbl_lbl_names = []
    for lbl in train_lbl_lbl:
        train_lbl_lbl_names.append(dict_lbl_idx_2_lbl[lbl])

    worst_augmentor = Augmentor(config, config.datapath, "WordReplacementLM", config.transform_times)
    list_worst_augmented_data = []
    for idx, txt in enumerate(train_lbl_txt):
        augmented_data_a, _, _ = worst_augmentor(txt, None, train_labeled_idxs[idx])
        list_worst_augmented_data.append(augmented_data_a)

    best_augmentor = Augmentor(config, config.datapath, "RandomDeletion", config.transform_times)
    list_best_augmented_data = []
    for idx, txt in enumerate(train_lbl_txt):
        augmented_data_a, _, _ = best_augmentor(txt[0], None, train_labeled_idxs[idx])
        list_best_augmented_data.append(augmented_data_a)

    top_fifty_txt = train_lbl_txt[:50]
    top_fifty_lbl = train_lbl_lbl_names[:50]
    top_fifty_best_aug = list_best_augmented_data[:50]
    top_fifty_worst_aug = list_worst_augmented_data[:50]

    with open("20_ng_augmentations.txt", 'w+') as f:
        for (txt, lbl, b_aug, w_aug) in zip(top_fifty_txt, top_fifty_lbl, top_fifty_best_aug, top_fifty_worst_aug):
            txt = txt[0]
            b_aug = b_aug[0]
            w_aug = w_aug[0][0]
            import ipdb; ipdb.set_trace()
            f.write('\t'.join([txt[0], lbl, b_aug[0], w_aug[0][0]]))


if __name__ == "__main__":
    main()