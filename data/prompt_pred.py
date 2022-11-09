import torch
import math
import numpy as np
import random
import argparse
import pickle
from tqdm import tqdm
from nltk.corpus import stopwords

from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from data.augmentation import convert_list_to_str
import logging

# To control logging level for various modules used in the application:
import logging
import re
def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)

def format_mrpc(train_labeled_data, num_examples):
    total_input = ""
    for datapoint in train_labeled_data[:num_examples]:
        input = datapoint[0]
        output = datapoint[1]
        if output == 0:
            prompt_template = f"{input[0]} is not semantically equivalent to {input[1]}."
        elif output == 1:
            prompt_template = f"{input[0]} is semantically equivalent to {input[1]}."
        else:
            raise ValueError("Invalid Output")

        total_input += " " + prompt_template
    new_example = train_labeled_data[num_examples]
    new_input = new_example[0]
    random_input = random.choice(new_input)
    new_output = new_example[1]
    if new_output == 0:
        total_input += f"{random_input} is not semantically equivalent to "
    elif new_output == 1:
        total_input += f"{random_input} is semantically equivalent to "
    else:
        raise ValueError("Invalid Output")
    return random_input, new_output, total_input

def prompt_pred(data_path, num_lbl, device):
    transformer = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    transformer.eval()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    with open(data_path + f'/train_{num_lbl}_labeled_data.pkl', 'rb') as f:
        train_labeled_data = list(pickle.load(f).values())

    all_prompt_input = []
    all_input = []
    for i in range(1000):
        random.shuffle(train_labeled_data)
        random_input, new_output, total_input = format_mrpc(train_labeled_data, 5)
        all_input.append(total_input)
        all_prompt_input.append((random_input, new_output))
    batch_size = 1
    list_augmentedData = []
    for start_idx in tqdm(range(0, 1000, batch_size)):
        batch_input = all_input[start_idx:min(1000, start_idx+batch_size)]
        tokenizer.pad_token = tokenizer.eos_token
        batch_dict = tokenizer(batch_input, return_tensors="pt", padding=True)
        input_ids = batch_dict["input_ids"].to(device)

        input_len = len(batch_dict["input_ids"][0])
        with torch.no_grad():

            batch_generatedIds = transformer.generate(input_ids,
                                                      attention_mask=batch_dict["attention_mask"].to(device),
                                                      do_sample=False,
                                                      max_length=50 + input_len)
        for idx, generated_ids in enumerate(batch_generatedIds):
            output = tokenizer.decode(batch_generatedIds[idx][input_len:], skip_special_tokens=True)
            list_augmentedData.append((all_prompt_input[start_idx * batch_size + idx], output))


    with open(data_path + '/few-shot_gen.pkl', 'wb') as f:
        pickle.dump(list_augmentedData, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', "--device", default=0, type=int)
    parser.add_argument('-n', "--num_label", default=10, type=int)

    parser.add_argument('-d', '--datapath', type=str, default='./processed_data/',
                        help='path to data folders')
    args = parser.parse_args()
    set_global_logging_level(logging.ERROR)

    device = torch.device("cuda:%d" % args.device if torch.cuda.is_available() else "cpu")
    random.seed(0)
    prompt_pred(args.datapath, args.num_label, device)







