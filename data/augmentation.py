import math
import random

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('words')
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import words


all_words = words.words()

from nltk.corpus import wordnet as wn

stopwords = stopwords.words('english')


def convert_list_to_str(list_wrd_idx):
    '''
    Convert list to store list in set
    :param list_wrd_idx:
    :return:
    '''
    return ' '.join(map(lambda x: str(x), list_wrd_idx))

def get_synonym(wrd, num_syn):
    '''
    Get synonyms of words
    :param wrd:
    :param num_syn:
    :return: set of synonyms
    '''
    set_aug = set()

    for ss in wn.synsets(wrd):
        for ssl in ss.lemma_names():

            if ssl != wrd:
                set_aug.add(ssl)
                if len(set_aug) >= num_syn:
                    return list(set_aug)

    return list(set_aug)

def _get_syn_per_wrd(list_wrds, depth):
    '''
    Get synonyms for every word
    :param list_wrds:
    :return:
    '''
    dict_idx2wrd_syn = {}

    for idx in range(len(list_wrds)):
        wrd = list_wrds[idx]
        list_synonyms = get_synonym(wrd, depth)

        if list_synonyms is None or len(list_synonyms) == 0:
            dict_idx2wrd_syn[idx] = None
        else:
            dict_idx2wrd_syn[idx] = list_synonyms

    return dict_idx2wrd_syn

def synonym_replacement(text, alpha, num_aug):
    '''
    Replace randomly sampled words with synonyms from NLTK

    :param text: text
    :param alpha: percentage of words to replace
    :param num_aug: number of augmentations
    :return:
    '''
    depth = 1 # number of synonyms per words
    num_perm_wrd = int(num_aug / depth) # number of permutations of indexes to flip

    list_wrds = word_tokenize(text)
    num_wrds = len(list_wrds)

    stop_wrd_idx = set(map(lambda x: x[0], filter(lambda x: x[1] in stopwords, enumerate(list_wrds))))
    dict_idx2wrd_syn = _get_syn_per_wrd(list_wrds, depth)
    # Filter our wrd_idx that do not have enough synonyms
    invalid_wrd_idx = set(filter(lambda idx: dict_idx2wrd_syn[idx] is None, range(num_wrds)))
    # Possible wrd_idx that can be flipped
    pos_flip_wrd_idx = list(set(range(num_wrds)).difference(stop_wrd_idx).difference(invalid_wrd_idx))

    num_wrd_flip = math.ceil(alpha * num_wrds)

    list_agmnt = []
    set_flip_wrd_idx = set()

    for _ in range(num_perm_wrd):
        # Ensure there are enough valid indices to flip
        if len(pos_flip_wrd_idx) < num_wrd_flip:
            if alpha > 0:
                # Recursively call the method with alpha lowered if not enough possible permutations
                return synonym_replacement(text, alpha-0.1, num_aug)
            else:
                # Return original text
                return [TreebankWordDetokenizer().detokenize(list_wrds), TreebankWordDetokenizer().detokenize(list_wrds)]

        # Keep sampling indices of words to flip until unique one comes up (at most 10 times)
        cntr = 0
        flip_wrd_idx = random.sample(pos_flip_wrd_idx, num_wrd_flip)
        while convert_list_to_str(flip_wrd_idx) in set_flip_wrd_idx:
            flip_wrd_idx = random.sample(pos_flip_wrd_idx, num_wrd_flip)
            cntr += 1
            # Note that if not enough words can be flipped, will return early with all possible augmentations but less than expected
            if cntr >= 10:
                if alpha > 0:
                    return synonym_replacement(text, alpha - 0.1, num_aug)
                else:
                    return [TreebankWordDetokenizer().detokenize(list_wrds),
                            TreebankWordDetokenizer().detokenize(list_wrds)]

        set_flip_wrd_idx.add(convert_list_to_str(flip_wrd_idx))

        # Generate num_syn augmentations by combining the ith top synonym per word to flip
        # Enumerating all possibilies is exponential
        for i in range(depth):
            agmnt_list_wrds = list_wrds.copy()
            for idx in flip_wrd_idx:
                list_syn = dict_idx2wrd_syn[idx]
                if i < len(list_syn):
                    agmnt_list_wrds[idx] = list_syn[i]
                else:
                    agmnt_list_wrds[idx] = random.choice(list_syn)
            list_agmnt.append(TreebankWordDetokenizer().detokenize(agmnt_list_wrds))

    return list_agmnt


def random_flip(text, alpha, num_aug):
    '''
    Randomly flip some words with words from NLTK list of words

    :param text: text
    :param alpha: percentage of words to replace
    :param num_aug: number of augmentations
    :return:
    '''
    depth = 1
    num_perm_wrd = int(num_aug / depth)

    list_wrds = list(word_tokenize(text))
    num_wrds = len(list_wrds)

    num_wrd_flip = math.ceil(alpha * num_wrds)

    list_agmnt = []
    set_flip_wrd_idx = set()

    for _ in range(num_perm_wrd):
        # Keep sampling indices of words to flip until unique one comes up (at most 10 times)
        cntr = 0

        flip_wrd_idx = random.sample(range(num_wrds), k=num_wrd_flip)
        while convert_list_to_str(flip_wrd_idx) in set_flip_wrd_idx:
            flip_wrd_idx = random.sample(range(num_wrds), k=num_wrd_flip)
            cntr += 1
            if cntr >= 10:
                # Note that if not enough words can be flipped, will return early with all possible augmentations but less than expected
                if alpha > 0:
                    return synonym_replacement(text, alpha - 0.1, num_aug)
                else:
                    return [TreebankWordDetokenizer().detokenize(list_wrds),
                            TreebankWordDetokenizer().detokenize(list_wrds)]

        set_flip_wrd_idx.add(convert_list_to_str(flip_wrd_idx))

        # Generate num_syn augmentations by combining the ith top synonym per word to flip
        # Enumerating all possibilies is exponential
        for i in range(depth):
            agmnt_list_wrds = list_wrds.copy()
            for idx in flip_wrd_idx:
                agmnt_list_wrds[idx] = random.choice(all_words)
            list_agmnt.append(TreebankWordDetokenizer().detokenize(agmnt_list_wrds))

    return list_agmnt

def random_insert(text, alpha, num_aug):
    '''
    Randomly insert words

    :param text: text
    :param alpha: percentage of words to replace
    :param num_aug: number of augmentations
    :return:
    '''
    depth = 1
    num_ins_wrd_idx = int(num_aug / depth)

    list_wrds = word_tokenize(text)
    num_wrds = len(list_wrds)

    num_wrd_ins = math.ceil(alpha * num_wrds)

    list_agmnt = []
    hash_agmnt = set()

    for _ in range(num_ins_wrd_idx):
        ins_wrd_idx = random.sample(range(num_wrds + num_wrd_ins), num_wrd_ins)
        ins_syn = random.choices(all_words, k=num_wrd_ins)
        hash_ins = convert_list_to_str(ins_wrd_idx) + convert_list_to_str(ins_syn)

        cntr = 0
        while hash_ins in hash_agmnt:
            ins_wrd_idx = random.sample(range(num_wrds + num_wrd_ins), num_wrd_ins)
            ins_syn = random.choices(all_words, k=num_wrd_ins)
            hash_ins = convert_list_to_str(ins_wrd_idx) + convert_list_to_str(ins_syn)
            cntr += 1
            if cntr >= 10:
                # Note that if not enough synonyms or places to insert new words, it will return early
                return list_agmnt

        hash_agmnt.add(hash_ins)

        agmnt_list_wrds = list_wrds.copy()
        for (idx, syn) in zip(ins_wrd_idx, ins_syn):
            agmnt_list_wrds.insert(idx, syn)

        list_agmnt.append(TreebankWordDetokenizer().detokenize(agmnt_list_wrds))

    return list_agmnt

def random_delete(text, alpha, num_aug):
    '''
    Randomly remove words

    :param text: text
    :param alpha: percentage of words to replace
    :param num_aug: number of augmentations
    :return:
    '''
    list_wrds = word_tokenize(text)
    hash_aug_text_seen = set()
    hash_aug_text_seen.add(convert_list_to_str(list_wrds))

    list_agmnt = []

    while len(list_agmnt) < num_aug:
        agmnt_list_wrds = []

        for wrd in list_wrds:
            if random.random() >= alpha:
                agmnt_list_wrds.append(wrd)

        hash_aug_text = convert_list_to_str(agmnt_list_wrds)
        if hash_aug_text not in hash_aug_text_seen:
            list_agmnt.append(TreebankWordDetokenizer().detokenize(agmnt_list_wrds))
        hash_aug_text_seen.add(hash_aug_text)

    return list_agmnt


def span_cutoff(text, alpha, num_aug):
    '''
    Randomly remove words

    :param text: text
    :param alpha: percentage of words to replace
    :param num_aug: number of augmentations
    :return:
    '''
    list_wrds = word_tokenize(text)
    num_wrds = len(list_wrds)

    len_span = int(num_wrds * alpha)

    list_agmnt = []

    while len(list_agmnt) < num_aug:
        agmnt_list_wrds = list_wrds.copy()
        start_idx = random.randint(0, num_wrds - len_span - 1)

        for i in range(start_idx, start_idx+len_span):
            agmnt_list_wrds[i] = "[PAD]"

        list_agmnt.append(TreebankWordDetokenizer().detokenize(agmnt_list_wrds))

    return list_agmnt

def word_flip(text, alpha, num_aug, set_words):
    '''
    Randomly flip words with other words from the dictionary

    :param text: text
    :param alpha: percentage of words to replace
    :param num_aug: number of augmentations
    :param set_words: set of words in dataset
    :return:
    '''
    depth = 1
    num_perm_wrd = int(num_aug / depth)

    list_wrds = list(word_tokenize(text))
    num_wrds = len(list_wrds)

    num_wrd_flip = math.ceil(alpha * num_wrds)

    list_agmnt = []
    set_flip_wrd_idx = set()

    for _ in range(num_perm_wrd):
        # Keep sampling indices of words to flip until unique one comes up (at most 10 times)
        cntr = 0

        flip_wrd_idx = random.sample(range(num_wrds), k=num_wrd_flip)
        while convert_list_to_str(flip_wrd_idx) in set_flip_wrd_idx:
            flip_wrd_idx = random.sample(range(num_wrds), k=num_wrd_flip)
            cntr += 1
            if cntr >= 10:
                # Note that if not enough words can be flipped, will return early with all possible augmentations but less than expected
                if alpha > 0:
                    return synonym_replacement(text, alpha - 0.1, num_aug)
                else:
                    return [TreebankWordDetokenizer().detokenize(list_wrds),
                            TreebankWordDetokenizer().detokenize(list_wrds)]

        set_flip_wrd_idx.add(convert_list_to_str(flip_wrd_idx))

        # Generate num_syn augmentations by combining the ith top synonym per word to flip
        # Enumerating all possibilies is exponential
        for i in range(depth):
            agmnt_list_wrds = list_wrds.copy()
            for idx in flip_wrd_idx:
                agmnt_list_wrds[idx] = random.choice(set_words)
            list_agmnt.append(TreebankWordDetokenizer().detokenize(agmnt_list_wrds))

    return list_agmnt

