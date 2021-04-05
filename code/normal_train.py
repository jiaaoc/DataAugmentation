import argparse
import os
import random
import math
import json
import contextlib


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





def main(config):
    set_seeds(config.seed)

    best_acc = 0

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
        train(labeled_trainloader, model, optimizer, criterion, epoch, config, n_labels)

        val_loss, val_acc, val_f1 = validate(config, 
            val_loader, model, criterion, epoch, mode='Valid Stats')

        print("epoch {}, val acc {}, val_loss {} val f1 {}".format(epoch, val_acc, val_loss, val_f1))
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

    print(best_acc)


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)

def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d

def train(labeled_trainloader, model, optimizer, criterion, epoch, config, n_labels):
    model.train()

    for batch_idx, (inputs, targets) in enumerate(labeled_trainloader):
        inputs = inputs.reshape(-1, config.max_seq_length)
        targets = targets.reshape(-1, )

        inputs, targets = inputs.to(config.device), targets.to(config.device)
        inputs = inputs.reshape(-1, inputs.shape[-1])
        targets = targets.reshape(-1)


        if config.emb_aug == "adv":
            # Copied from https://github.com/lyakaap/VAT-pytorch

            with torch.no_grad():
                # outputs = F.softmax(model(inputs), dim=-1)  # [bs, num_classes]

                input_emb = model.get_embedding_output(inputs)
                outputs = F.softmax(model.get_bert_output(input_emb), dim=-1)

            d = torch.rand(input_emb.shape).sub(0.5).to(input_emb.device)
            d = _l2_normalize(d)

            with _disable_tracking_bn_stats(model):
                # calc adversarial direction
                for _ in range(1):
                    d.requires_grad_()
                    pred_hat = model.get_bert_output(input_emb + 10 * d)
                    logp_hat = F.log_softmax(pred_hat, dim=1)
                    adv_distance = F.kl_div(logp_hat, outputs, reduction='batchmean')
                    adv_distance.backward()
                    d = _l2_normalize(d.grad)
                    model.zero_grad()

                # calc LDS
                r_adv = d * 1
                pred_hat = model.get_bert_output(input_emb + r_adv)
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_loss = F.kl_div(logp_hat, outputs, reduction='batchmean')

            outputs = model(inputs) # [bs, num_classes]
            normal_loss = criterion(outputs, targets)
            loss = config.lambda_u * adv_loss + normal_loss

        elif config.emb_aug == "mixup":



            idx = torch.randperm(inputs.size(0))
            input_a, input_b = inputs, inputs[idx]

            targets_emb = torch.zeros(input_a.shape[0], n_labels).scatter_(torch.tensor(1).to(targets.device), targets.view(-1, 1), torch.tensor(1).to(targets.device))

            target_a, target_b = targets_emb, targets_emb[idx]


            input_a_emb = model.get_embedding_output(input_a)
            input_b_emb = model.get_embedding_output(input_b)

            l = np.random.beta(1, 1)

            new_input = input_a_emb * l + (1-l) * input_b_emb
            new_target = target_a * l + (1-l) * target_b
            new_output = model.get_bert_output(new_input)

            mixup_loss = F.kl_div(new_output, new_target, reduction='batchmean')

            outputs = model(inputs) # [bs, num_classes]
            normal_loss = criterion(outputs, targets)
            loss = config.lambda_u * mixup_loss + normal_loss

        else:
            outputs = model(inputs) # [bs, num_classes]
            loss = criterion(outputs, targets)

        print('epoch {}, step {}, loss {}'.format(
            epoch, batch_idx, loss.item()))
        print("Finished %d" % batch_idx, end='\r')

        #if  config.grad_accumulation_factor > 1:
        loss = loss /  config.grad_accumulation_factor

        loss.backward()
        
        if (batch_idx+1) % config.grad_accumulation_factor == 0:
            optimizer.step()
            optimizer.zero_grad()

        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Data Augmentation')
    parser.add_argument('-c', '--config_file', required=True)
    parser.add_argument('-k', '--kwargs', nargs='*', action=ParseKwargs)

    args = parser.parse_args()

    config = Config(args.config_file, args.kwargs)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device

    main(config)
