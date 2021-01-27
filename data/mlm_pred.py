import torch
import math
import numpy as np
import random
import argparse
import pickle
from tqdm import tqdm
from nltk.corpus import stopwords

from transformers import AutoModelWithLMHead, AutoTokenizer
from data.augmentation import convert_list_to_str




def mlm_pred(model, tokenizer, masked_text, num_mask, top_k):
    '''
    :param model:
    :param tokenizer:
    :param masked_text:
    :param masked_idx:
    :param num_mask:
    :param top_k:
    :return:
    '''

    bs = len(masked_text)
    input = tokenizer.batch_encode_plus(masked_text, return_tensors="pt", padding=True, truncation=True)['input_ids']
    token_logits = model(input.to(device))[0].detach().cpu() # [bs, max_len, num_vocab]
    num_vocab = token_logits.shape[2]
    mask_token_index = torch.where(input == tokenizer.mask_token_id)  # ([bs * num_mask], [bs * num_mask])

    num_mask = mask_token_index[0].shape[0]

    mask_token_index_reshaped = mask_token_index[1].reshape(bs, num_mask, 1).repeat(1, 1, num_vocab) # [bs, num_mask, num_vocab]
    mask_token_logits = torch.gather(token_logits, 1, mask_token_index_reshaped) # [bs, num_mask, num_vocab]
    top_k_tokens = torch.topk(mask_token_logits, top_k, dim=2).indices.reshape(-1)  # [bs, num_mask, top_k]
    wrds = tokenizer.convert_ids_to_tokens(top_k_tokens)
    wrds = np.asarray(wrds).reshape(bs, num_mask, top_k).transpose(0, 2, 1) # [bs, top_k, mask]
    return wrds

def augment_single_text_mlm_flip(model, tokenizer, stop_words, alpha, num_syn, num_flip_wrd_idx, text):
    '''
    Augment single txt
    :param text:
    :param num_aug:
    :return:
    '''

    list_wrds = text.strip('\n').split(' ')
    num_wrds = min(len(list_wrds), 512)
    num_wrd_flip = int(math.ceil(alpha * num_wrds))

    stop_wrd_idx = set(map(lambda x: x[0], filter(lambda x: x[1] in stop_words, enumerate(list_wrds))))
    # Ignore stop word to flip
    pos_flip_wrd_idx = list(set(range(num_wrds)).difference(stop_wrd_idx))

    list_agmnt = []
    seen_flip_wrd_idx = set()

    for _ in range(num_flip_wrd_idx):
        # Ensure there are enough valid indices to flip
        if len(pos_flip_wrd_idx) < num_wrd_flip:
            return list_agmnt

        # Keep sampling indices of words to flip until unique one comes up (at most 10 times)
        cntr = 0
        flip_wrd_idx = random.sample(pos_flip_wrd_idx, num_wrd_flip)
        while convert_list_to_str(flip_wrd_idx) in seen_flip_wrd_idx:
            flip_wrd_idx = random.sample(pos_flip_wrd_idx, num_wrd_flip)
            cntr += 1
            if cntr >= 10:
                # Note that if not enough words can be flipped, will return early with all possible augmentations but less than expected
                return list_agmnt

        if len(flip_wrd_idx) > num_wrd_flip:
            flip_wrd_idx = flip_wrd_idx[:num_wrd_flip]

        seen_flip_wrd_idx.add(convert_list_to_str(flip_wrd_idx))
        agmnt_list_wrds = list_wrds.copy()
        for idx in flip_wrd_idx:
            agmnt_list_wrds[idx] = "[MASK]"

        num_syn_pred_wrds = mlm_pred(model, tokenizer, [" ".join(agmnt_list_wrds)], num_wrd_flip, int(num_syn))[0]

        # Generate num_syn augmentations by combining the ith top synonym per word to flip
        # Enumerating all possibilies is exponential
        for i in range(num_syn):
            agmnt_list_wrds = list_wrds.copy()
            cur_pred_wrd_idx = 0
            for idx in flip_wrd_idx:
                agmnt_list_wrds[idx] = num_syn_pred_wrds[i][cur_pred_wrd_idx]
                cur_pred_wrd_idx += 1
                if cur_pred_wrd_idx < len(num_syn_pred_wrds[i]):
                    break
            list_agmnt.append(" ".join(agmnt_list_wrds))

    return list_agmnt


def all_mlm_pred(data_path, device):
    model = AutoModelWithLMHead.from_pretrained("bert-base-uncased").to(device)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    stop_words = stopwords.words('english')

    with open(data_path + 'train_unlabeled_data.pkl', 'rb') as f:
        train_unlabeled_data = pickle.load(f)

    cnt = 0
    train_unlabeled_data_aug = {}
    for key, value in tqdm(train_unlabeled_data.items(), ncols=50, desc="Iteration:"):
        if isinstance(value, list):
            new_value = []
            for text in value:
                new_text = augment_single_text_mlm_flip(model, tokenizer, stop_words, 0.1, 1, 2, text)
                new_value.append(new_text)

            train_unlabeled_data_aug[key] = new_value

        else:
            if len(value) == 2:
                new_value = augment_single_text_mlm_flip(model, tokenizer, stop_words, 0.1, 1, 2, value)
                train_unlabeled_data_aug[key] = new_value

        cnt += 1

        if cnt % 5000 == 0:
            with open(data_path + 'mlm.pkl', 'wb') as f:
                pickle.dump(train_unlabeled_data_aug, f)

    with open(data_path + 'mlm.pkl', 'wb') as f:
        pickle.dump(train_unlabeled_data_aug, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', "--device", default=0, type=int)
    parser.add_argument('-d', '--datapath', type=str, default='./processed_data/',
                        help='path to data folders')
    args = parser.parse_args()

    device = torch.device("cuda:%d" % args.device if torch.cuda.is_available() else "cpu")

    all_mlm_pred(args.datapath, device)







