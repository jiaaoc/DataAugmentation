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

def format_qqp(train_labeled_data, num_examples):
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

def format_qnli(train_labeled_data, num_examples):
    total_input = ""
    for datapoint in train_labeled_data[:num_examples]:
        input = datapoint[0]
        output = datapoint[1]
        if output == 0:
            prompt_template = f"{input[0]} does not imply {input[1]}."
        elif output == 1:
            prompt_template = f"{input[0]} implies {input[1]}."
        else:
            raise ValueError(f"Invalid Output {output}")

        total_input += " " + prompt_template
    new_example = train_labeled_data[num_examples]
    new_input = new_example[0]
    new_output = new_example[1]
    if new_output == 0:
        total_input += f"{new_input[0]} does not imply"
    elif new_output == 1:
        total_input += f"{new_input[0]} implies "
    else:
        raise ValueError("Invalid Output")
    return new_input[0], new_output, total_input

def format_20ng(train_labeled_data, num_examples):
    total_input = ""
    dict_labelIdx_toName = {
        0: "atheism",
        1: "computer graphics",
        2: "computer operating system microsoft windows",
        3: "computer system ibm hardware",
        4: "computer system mac hardware",
        5: "computer windows x",
        6: "items for sale",
        7: "recreation autos",
        8: "recreation motorcycles",
        9: "recreation sports baseball",
        10: "recreation sports hockey",
        11: "science cryptography",
        12: "science electronics",
        13: "science medicine",
        14: "science space",
        15: "religion christian",
        16: "politics guns",
        17: "politics middle east",
        18: "politics",
        19: "religion"
    }

    for datapoint in train_labeled_data[:num_examples]:
        input = datapoint[0]
        output = datapoint[1]
        prompt_template = f"The follow email is about {dict_labelIdx_toName[int(output)]}: {input}"

        total_input += " " + prompt_template
    new_example = train_labeled_data[num_examples]
    new_input = new_example[0]
    new_output = new_example[1]
    total_input += f"The follow email is about {dict_labelIdx_toName[int(new_output)]}:"
    return new_input[0], new_output, total_input

def format_pubmed(train_labeled_data, num_examples):
    total_input = ""
    dict_labelIdx_toName = {
        0: "objective",
        1: "methods",
        2: "results",
        3: "conclusion",
        4: "background"
    }
    for datapoint in train_labeled_data[:num_examples]:
        input = datapoint[0]
        output = datapoint[1]
        prompt_template = f"The follow sentence in the abstract is about {dict_labelIdx_toName[int(output)]}: {input}"

        total_input += " " + prompt_template
    new_example = train_labeled_data[num_examples]
    new_input = new_example[0]
    new_output = new_example[1]
    total_input += f"The follow sentence in the abstract is about {dict_labelIdx_toName[int(new_output)]}: "
    return new_input[0], new_output, total_input

def format_mnli(train_labeled_data, num_examples):
    total_input = ""
    for datapoint in train_labeled_data[:num_examples]:
        input = datapoint[0]
        output = datapoint[1]
        if output == 0:
            prompt_template = f"{input[0]} does not imply {input[1]}."
        elif output == 1:
            prompt_template = f"{input[0]} is neutral with {input[1]}."
        elif output == 2:
            prompt_template = f"{input[0]} implies {input[1]}."
        else:
            raise ValueError("Invalid Output")

        total_input += " " + prompt_template
    new_example = train_labeled_data[num_examples]
    new_input = new_example[0]
    new_output = new_example[1]
    if new_output == 0:
        total_input += f"{new_input[0]} does not imply"
    elif new_output == 1:
        total_input += f"{new_input[0]} is neutral with "
    elif new_output == 2:
        total_input += f"{new_input[0]} implies "
    else:
        raise ValueError("Invalid Output")
    return new_input[0], new_output, total_input

def prompt_pred(data_path, dataset, num_lbl, device):
    transformer = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    transformer.eval()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    with open(data_path + f'/train_{num_lbl}_labeled_data.pkl', 'rb') as f:
        train_labeled_data = list(pickle.load(f).values())

    all_prompt_input = []
    all_input = []
    for i in range(1000):
        random.shuffle(train_labeled_data)
        if dataset == "mrpc":
            random_input, new_output, total_input = format_mrpc(train_labeled_data, 5)
        elif dataset == "qqp":
            random_input, new_output, total_input = format_qqp(train_labeled_data, 5)
        elif dataset == "qnli":
            random_input, new_output, total_input = format_qnli(train_labeled_data, 5)
        elif dataset == "mnli":
            random_input, new_output, total_input = format_mnli(train_labeled_data, 5)
        elif dataset == "pubmed":
            random_input, new_output, total_input = format_pubmed(train_labeled_data, 5)
        elif dataset == "20_ng":
            random_input, new_output, total_input = format_20ng(train_labeled_data, 5)

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


    with open(data_path + f'/few-shot_gen_{num_lbl}_lbl.pkl', 'wb') as f:
        pickle.dump(list_augmentedData, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', "--device", default=0, type=int)
    parser.add_argument('-n', "--num_label", default=10, type=int)
    parser.add_argument("--dataset", choices=["mrpc", "qqp", "qnli", "mnli", "pubmed",
                                                      "20_ng"])
    parser.add_argument('-d', '--datapath', type=str, default='./processed_data/',
                        help='path to data folders')
    args = parser.parse_args()
    set_global_logging_level(logging.ERROR)

    device = torch.device("cuda:%d" % args.device if torch.cuda.is_available() else "cpu")
    random.seed(0)
    prompt_pred(args.datapath, args.dataset, args.num_label, device)







