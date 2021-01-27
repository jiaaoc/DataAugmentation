import torch
import torch.nn as nn
import argparse
from code.CLS_model import CLS_model
from code.Config import Config
import torch.utils.data as Data
import json

from code.train import validate
from code.util import ParseKwargs
from code.read_data import get_data

parser = argparse.ArgumentParser(description='PyTorch Data Augmentation')
parser.add_argument('-c', '--config_file', required=True)
parser.add_argument('-k', '--kwargs', nargs='*', action=ParseKwargs)
args = parser.parse_args()

config = Config(args.config_file, args.kwargs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(config):

    _, _, _, test_set, n_labels = get_data(config)

    test_loader = Data.DataLoader(
        dataset=test_set, batch_size=config.test_batch_size, shuffle=False)

    model = CLS_model(config, n_labels).to(device)

    criterion = nn.CrossEntropyLoss()

    model.load_state_dict(torch.load(config.best_model_file))
    test_loss, test_acc, test_f1 = validate(
        config, test_loader, model, criterion, config.epoch, mode='Test Stats ')
    with open(config.test_score_file, 'a+') as f:
        f.write(json.dumps({"epoch": config.epoch, "best_test_acc": test_acc, "best_test_f1": test_f1}) + '\n')


if __name__ == "__main__":
    test(config)