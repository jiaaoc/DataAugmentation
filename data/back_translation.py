import torch
#import fairseq
import argparse
import pickle
import os
import logging
from tqdm import tqdm

from transformers import FSMTForConditionalGeneration, FSMTTokenizer
mname = "facebook/wmt19-en-de"
tokenizer = FSMTTokenizer.from_pretrained(mname)

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description='back translation')
parser.add_argument('--gpu', default='6', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--data_path', type=str, default='./processed_data/',
                    help='path to data folders')

args = parser.parse_args()

# List available models
print(torch.hub.list('pytorch/fairseq'))  # [..., 'transformer.wmt16.en-de', ... ]

# Load a transformer trained on WMT'16 En-De
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de',  checkpoint_file='model1.pt',
                       tokenizer='moses', bpe='fastbpe')
de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en',  checkpoint_file='model1.pt',
                       tokenizer='moses', bpe='fastbpe')

#assert isinstance(en2de.models[0], fairseq.models.transformer.TransformerModel)

data_path = args.data_path

with open(data_path + 'train_unlabeled_data.pkl', 'rb') as f:
    train_unlabeled_data = pickle.load(f)

num_sample_sen = 2
cnt = 0
train_unlabeled_data_aug = {}
gpu = args.gpu
os.environ['CUDA_VISIBLE_DEVICES'] = gpu
en2de = en2de.cuda()
de2en = de2en.cuda()

for key, value in tqdm(train_unlabeled_data.items(), ncols=50, desc="Iteration:"):
    new_value = []

    input_ids = tokenizer(value)["input_ids"]
    trimmed_input_ids = input_ids[:256]
    trimmed_value = tokenizer.decode(trimmed_input_ids)

    for i in range(num_sample_sen):
        sample = en2de.translate(trimmed_value, sampling = True, temperature = 0.8,  skip_invalid_size_inputs=True)
        v = de2en.translate(sample, sampling = True, temperature = 0.8, skip_invalid_size_inputs=True)
        if cnt % 100 == 0:
            print("***************")
        new_value.append(v)
    train_unlabeled_data_aug[key] = new_value
    if cnt % 1000 == 0:
        with open(data_path + 'train_unlabeled_data_bt.pkl', 'wb') as f:
            assert len(train_unlabeled_data_aug[key]) == num_sample_sen
            pickle.dump(train_unlabeled_data_aug, f)
    cnt += 1

with open(data_path + 'train_unlabeled_data_bt.pkl', 'wb') as f:
    pickle.dump(train_unlabeled_data_aug, f)