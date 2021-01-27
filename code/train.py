import argparse
import os
import random
import math
import argparse
import logging

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from transformers import *
from sklearn.metrics import f1_score


from code.CLS_model import CLS_model
from torch.autograd import Variable
from torch.utils.data import Dataset
import json
from code.read_data import *

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
n_gpu = torch.cuda.device_count()

best_acc = 0

def main(config):
    set_seeds(config.seed)

    global best_acc
    # Read dataset and build dataloaders
    train_labeled_set, train_unlabeled_set, val_set, test_set, n_labels = get_data(config)

    labeled_trainloader = Data.DataLoader(
        dataset=train_labeled_set, batch_size=config.batch_size, shuffle=True)
    unlabeled_trainloader = Data.DataLoader(
        dataset=train_unlabeled_set, batch_size=config.batch_size_u, shuffle=True)
    val_loader = Data.DataLoader(
        dataset=val_set, batch_size=config.test_batch_size, shuffle=False)
    test_loader = Data.DataLoader(
        dataset=test_set, batch_size=config.test_batch_size, shuffle=False)


    #TODO: get number of labels
    model = CLS_model(config, n_labels).to(device)

    #
    # if config.n_gpu > 1:
    #     model = nn.DataParallel(model)

    optimizer = AdamW(model.parameters(), lr=config.lr)

    criterion = nn.CrossEntropyLoss()

    f = open(config.dev_score_file, 'w')

    # Start training
    for epoch in range(config.epochs):
        train(labeled_trainloader, unlabeled_trainloader, model, optimizer,
            epoch, n_labels, config)

        val_loss, val_acc, val_f1 = validate(
            config, val_loader, model, criterion, epoch, mode='Valid Stats')

        print("epoch {}, val acc {}, val_loss {}".format(epoch, val_acc, val_loss))

        with open(config.dev_score_file, 'a+') as f:
            f.write(json.dumps({"epoch": epoch, "val_acc": val_acc, "val_f1": val_f1}) + '\n')

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), config.best_model_file)


    model.load_state_dict(torch.load(config.best_model_file))
    test_loss, test_acc, test_f1 = validate(
        config, test_loader, model, criterion, epoch, mode='Test Stats ')
    with open(config.test_score_file, 'a+') as f:
        f.write(json.dumps({"epoch": epoch, "best_test_acc": test_acc, "best_test_f1": test_f1}) + '\n')


def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, epoch, n_labels, config):
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    model.train()

    for batch_idx in range(config.val_iteration):
        try:
            inputs_x, targets_x = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = labeled_train_iter.next()

        try:
            (inputs_ori, inputs_u) = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_ori, inputs_u) = unlabeled_train_iter.next()

        inputs_x = inputs_x.reshape(-1, config.max_seq_length)
        targets_x = targets_x.reshape(-1)

        #print(inputs_x.shape, inputs_u.shape, inputs_ori.shape)

        batch_size = inputs_x.size(0)
        # inputs_x = torch.tensor(inputs_x[:,0]) # [bs * (1+num_aug), max_seq_len]
        # targets_x = torch.tensor(targets_x[:,0]) # [bs * (1+num_aug)]

        targets_x = torch.zeros(batch_size, n_labels).scatter_(
            1, targets_x.view(-1, 1), 1) #[]

        inputs_x, targets_x = inputs_x.to(device), targets_x.to(device)

        inputs_u = inputs_u.to(device)
        inputs_ori = inputs_ori.to(device)

        with torch.no_grad():
            # Predict labels for unlabeled data.
            outputs_ori = model(inputs_ori)

            sharp_outputs_ori_prob = F.softmax(outputs_ori / config.sharp_temperature)

        all_inputs = torch.cat([inputs_x, inputs_u], dim=0) # [bs+bs_u]

        logits = model(all_inputs)

        cur_step = epoch+batch_idx/config.val_iteration

        tsa_thresh = get_tsa_thresh("linear_schedule", cur_step, config.epochs, 0, 1/ n_labels, device)

        
        outputs_x = logits[:batch_size]
        outputs_u = logits[batch_size:]


        sup_loss = torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1)  # [bs, ]
        
        less_than_threshold = torch.exp(sup_loss) < tsa_thresh  # prob = exp(log_prob), prob > tsa_threshold
        
        Lx = torch.sum(sup_loss * less_than_threshold, dim=-1) / torch.max(torch.sum(less_than_threshold, dim=-1),
                                                                           torch.tensor(1.).to(device).long())

        probs_u = torch.log_softmax(outputs_u, dim=1)

        Lu = F.kl_div(probs_u, sharp_outputs_ori_prob, reduction='batchmean')  # [bs, ]
        

        print("epoch {}, step {}, Lx {}, Lu {}".format(
                epoch, batch_idx, Lx.item(), Lu.item()))

        loss = Lx + config.lambda_u * Lu
        loss += loss / config.grad_accumulation_factor
        loss.backward()

        if (batch_idx+1) % config.grad_accumulation_factor == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        if batch_idx % 400 == 0:
            print("epoch {}, step {}, loss {}, Lx {}, Lu {}".format(
                epoch, batch_idx, loss.item(), Lx.item(), Lu.item()))

def validate(config, valloader, model, criterion, epoch, mode):
    model.eval()
    with torch.no_grad():
        loss_total = 0
        total_sample = 0
        acc_total = 0
        correct = 0
        pred_lbl = []
        true_lbl = []

        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            targets = targets.reshape(-1)
            loss = criterion(outputs, targets)

            #pred_lbl.extend(np.array(predicted.cpu()).tolist())
            #true_lbl.extend(np.array(targets.cpu()).tolist())

            if config.is_classification:
                _, predicted = torch.max(outputs.data, 1)


                pred_lbl.extend(np.array(predicted.cpu()).tolist())
                true_lbl.extend(np.array(targets.cpu()).tolist())

                correct += (np.array(predicted.cpu()) ==
                            np.array(targets.cpu())).sum()
                loss_total += loss.item() * inputs.shape[0]
                total_sample += inputs.shape[0]


        if config.is_classification:
            f1 = f1_score(true_lbl, pred_lbl, average=None)
            avg_f1 = np.mean(f1)

            acc_total = correct/total_sample
            loss_total = loss_total/total_sample

            return loss_total, acc_total, avg_f1

        else:
            if "stsb" in config.dataset.lower():
                pearson_corr = pearsonr(true_lbl, pred_lbl)[0]

                return loss_total, pearson_corr, 0
            else:
                matthews_cor = matthews_corrcoef(true_lbl, pred_lbl)[0]
                return loss_total, matthews_cor, 0

def get_tsa_thresh(schedule, global_step, num_train_steps, start, end, device):
    training_progress = torch.tensor(float(global_step) / float(num_train_steps))
    if schedule == 'linear_schedule':
        threshold = training_progress
    elif schedule == 'exp_schedule':
        scale = 5
        threshold = torch.exp((training_progress - 1) * scale)
    elif schedule == 'log_schedule':
        scale = 5
        threshold = 1 - torch.exp((-training_progress) * scale)
    output = threshold * (end - start) + start
    return output.to(device)

if __name__ == '__main__':



    

    main(config)
