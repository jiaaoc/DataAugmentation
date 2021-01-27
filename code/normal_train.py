import argparse
import os
import random
import math
import json


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
# from pytorch_transformers import *
from torch.autograd import Variable
from torch.utils.data import Dataset

from code.CLS_model import CLS_model
from code.read_data import *
from code.normal_bert import ClassificationBert
from code.train import validate

from code.util import ParseKwargs
from code.Config import Config
from code.util import set_seeds



parser = argparse.ArgumentParser(description='PyTorch Data Augmentation')
parser.add_argument('-c', '--config_file', required=True)
parser.add_argument('-k', '--kwargs', nargs='*', action=ParseKwargs)

args = parser.parse_args()

config = Config(args.config_file, args.kwargs)

os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_acc = 0

def main(config):
    set_seeds(config.seed)

    global best_acc

    train_labeled_set, train_unlabeled_set, val_set, test_set, n_labels = get_data(config)
    labeled_trainloader = Data.DataLoader(
        dataset=train_labeled_set, batch_size=config.batch_size, shuffle=True)
    val_loader = Data.DataLoader(
        dataset=val_set, batch_size=config.test_batch_size, shuffle=False)
    test_loader = Data.DataLoader(
        dataset=test_set, batch_size=config.test_batch_size, shuffle=False)

    model = CLS_model(config, n_labels).to(device)

    # if args.n_gpu > 1:
    #     model = nn.DataParallel(model)
    
    optimizer = AdamW(model.parameters(), lr=config.lr)

    criterion = nn.CrossEntropyLoss()

    #with open(config.dev_score_file, 'w') as f:
    #    pass
    f = open(config.dev_score_file, 'w')

    for epoch in range(config.epochs):
        train(labeled_trainloader, model, optimizer, criterion, epoch, config)

        val_loss, val_acc, val_f1 = validate(config, 
            val_loader, model, criterion, epoch, mode='Valid Stats')

        print("epoch {}, val acc {}, val_loss {} val f1 {}".format(epoch, val_acc, val_loss, val_f1))
        with open(config.dev_score_file, 'a+') as f:
            f.write(json.dumps({"epoch": epoch, "val_acc": val_acc, "val_f1": val_f1}) + '\n')
            #json.dump({"epoch": epoch, "val_acc": val_acc, "val_f1": val_f1}, f)

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), config.best_model_file)

        print('Epoch: ', epoch)

    model.load_state_dict(torch.load(config.best_model_file))
    test_loss, test_acc, test_f1 = validate(
        test_loader, model, criterion, epoch, mode='Test Stats ')

    with open(config.dev_score_file, 'a+') as f:
        f.write(json.dumps({"epoch": epoch, "best_test_acc": test_acc}) + '\n')
        #json.dump({"epoch": epoch, "val_acc": val_acc, "val_f1": val_f1}, f)

    print('Best acc:')
    print(best_acc)


    # logger.info("******Finished training, test acc {}******".format(test_accs[-1]))
    print("Finished training!")
    print('Best acc:')
    print(best_acc)



def train(labeled_trainloader, model, optimizer, criterion, epoch, config):
    model.train()

    for batch_idx, (inputs, targets) in enumerate(labeled_trainloader):
        inputs = inputs.reshape(-1, config.max_seq_length)
        targets = targets.reshape(-1, )
        #print(inputs.shape, targets.shape)
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.reshape(-1, inputs.shape[-1])
        targets = targets.reshape(-1)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        print('epoch {}, step {}, loss {}'.format(epoch, batch_idx, loss.item()))

        #if  config.grad_accumulation_factor > 1:
        loss = loss /  config.grad_accumulation_factor

        loss.backward()
        
        if (batch_idx+1) % config.grad_accumulation_factor == 0:
            optimizer.step()
            optimizer.zero_grad()

        


if __name__ == '__main__':
    

    main(config)
