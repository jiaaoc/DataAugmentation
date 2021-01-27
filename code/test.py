import torch
import torch.nn as nn
import argparse
from code.CLS_model import CLS_model
from code.Config import Config
import torch.utils.data as Data
import json
import os

from code.train import validate
from code.util import ParseKwargs
from code.read_data import get_data



def test(config):

    _, _, _, test_set, n_labels = get_data(config)

    test_loader = Data.DataLoader(
        dataset=test_set, batch_size=config.test_batch_size, shuffle=False)

    model = CLS_model(config, n_labels).to(device)

    criterion = nn.CrossEntropyLoss()

    model.load_state_dict(torch.load(config.best_model_file))
    test_loss, test_acc, test_f1 = validate(
        config, test_loader, model, criterion, config.epochs, mode='Test Stats ')
    with open(config.test_score_file, 'a+') as f:
        f.write(json.dumps({"epoch": config.epochs, "best_test_acc": test_acc, "best_test_f1": test_f1}) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp_dir', required=True)
    args = parser.parse_args()

    config = Config(os.path.join(args.exp_dir, 'config.json'), {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device
    test(config)