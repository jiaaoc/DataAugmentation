import argparse
import os
import random
import math
import json

# import logging

# from transformers.utils.logging import logging

from sklearn.metrics import f1_score

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
# from pytorch_transformers import *
from torch.autograd import Variable
from torch.utils.data import Dataset

from code.read_data import *
from code.normal_bert import ClassificationBert

# logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description='PyTorch Data Augmentation')


parser.add_argument('--gpu', default='0,1,2,3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')


# Training parameters
parser.add_argument('--model', type=str, default='bert-base-uncased',
                    help='pretrained model')

parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=4, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch-size', default=24, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--batch-size-u', default=24, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--val-iteration', type=int, default=200,
                    help='frequency of evaluation')

parser.add_argument('--lrmain', '--learning-rate-bert', default=0.00001, type=float,
                    metavar='LR', help='initial learning rate for bert')
parser.add_argument('--lrlast', '--learning-rate-model', default=0.001, type=float,
                    metavar='LR', help='initial learning rate for models')

parser.add_argument('--lambda-u', default=1, type=float,
                    help='weight for consistency loss term of unlabeled data')

parser.add_argument('--T', default=0.5, type=float,
                    help='temperature for sharpen function')

parser.add_argument('--temp-change', default=1000000, type=int)


# Datasets
parser.add_argument('--data-path', type=str, default='yahoo_answers_csv/',
                    help='path to data folders')

parser.add_argument('--n-labeled', type=int, default=10,
                    help='number of labeled data')

parser.add_argument('--un-labeled', default=5000, type=int,
                    help='number of unlabeled data')

parser.add_argument('--output-dir', default="test_model", type=str,
                    help='path to trained model and eval and test results')


# Augmentations
parser.add_argument('--train-aug', default=False, type=bool, metavar='N',
                    help='augment labeled training data')

parser.add_argument('--transform-type', type=str, default=None,
                    help='augmentation type')

parser.add_argument('--transform-times', default=1, type=int,
                    help='number of augmentations per sample')


args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
args.n_gpu = n_gpu

# logger.info("Training/evaluation parameters %s", args)

best_acc = 0
total_steps = 0
flag = 0


def main():
    global best_acc

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)


    # logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    #                     datefmt='%m/%d/%Y %H:%M:%S',
    #                     level=logging.INFO)
    # logger.warning("Device: %s, n_gpu: %s", device, args.n_gpu)

    train_labeled_set, train_unlabeled_set, val_set, test_set, n_labels = get_data(
        args.data_path, args.n_labeled, args.un_labeled, model=args.model, train_aug=args.train_aug,
        transform_type=args.transform_type, transform_times = args.transform_times)

    labeled_trainloader = Data.DataLoader(
        dataset=train_labeled_set, batch_size=args.batch_size, shuffle=True)
    # unlabeled_trainloader = Data.DataLoader(
    #     dataset=train_unlabeled_set, batch_size=args.batch_size_u, shuffle=True)
    val_loader = Data.DataLoader(
        dataset=val_set, batch_size=args.test_batch_size, shuffle=False)
    test_loader = Data.DataLoader(
        dataset=test_set, batch_size=args.test_batch_size, shuffle=False)


    model = ClassificationBert(n_labels).cuda()
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    
    optimizer = AdamW(
        [
            {"params": model.bert.parameters(), "lr": args.lrmain},
            {"params": model.linear.parameters(), "lr": args.lrlast},
        ])

    # train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss()

    test_accs = []

    underscore_data_path = os.path.split(os.path.split(args.data_path)[0])[1].replace("/", "_")
    file_name = "_".join([underscore_data_path, str(args.n_labeled), str(args.un_labeled), str(args.transform_type)])

    data_directory = os.path.join("exp_out", underscore_data_path)

    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    file = os.path.join(data_directory, file_name)

    f =  open(file, 'w+')

    # logger.info("***** Running training *****")
    # logger.info("  Num labeled examples = %d", len(labeled_trainloader))
    # # logger.info("  Num unlabeled examples = %d", len(unlabeled_trainloader))
    # logger.info("  Num Epochs = %d", args.epochs)
    # logger.info("  Lambda_u = %s" % str(args.lambda_u))
    # logger.info("  Batch size = %d" % args.batch_size)
    # logger.info("  Max seq length = %d" % 256)

    for epoch in range(args.epochs):
        # train(labeled_trainloader, unlabeled_trainloader, model, optimizer,
        #       scheduler, train_criterion, epoch, n_labels, args.train_aug)
        train(labeled_trainloader, model, optimizer, criterion, epoch)

        val_loss, val_acc, val_f1 = validate(
            val_loader, model, criterion, epoch, mode='Valid Stats')
        logger.info("******Epoch {}, val acc {}, val loss {}, val f1 {} ******".format(epoch, val_acc, val_loss, val_f1))

        print("epoch {}, val acc {}, val_loss {} val f1 {}".format(
            epoch, val_acc, val_loss, val_f1))
        f.write(json.dumps({"epoch": epoch, "val_acc": val_acc, "val_f1": val_f1}) + '\n')

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), file + ".pt")
            # test_loss, test_acc, test_f1 = validate(
            #     test_loader, model, criterion, epoch, mode='Test Stats ')
            # f.write(json.dumps({"epoch": epoch, "best_test_acc": test_acc, "best_test_f1": test_f1}) + '\n')
            # test_accs.append(test_acc)
            # # logger.info("******Epoch {}, test acc {}, test loss {}******".format(epoch, test_acc, test_loss))
            # print("epoch {}, test acc {}, test f1 {}, test loss {}".format(
            #     epoch, test_acc, test_f1, test_loss))

        print('Epoch: ', epoch)

    model.load_state_dict(torch.load(file + ".pt"))
    test_loss, test_acc, test_f1 = validate(
        test_loader, model, criterion, epoch, mode='Test Stats ')
    f.write(json.dumps({"epoch": epoch, "best_test_acc": test_acc, "best_test_f1": test_f1}) + '\n')
    test_accs.append(test_acc)

    print('Best acc:')
    print(best_acc)

    print('Test acc:')
    print(test_accs)


    # logger.info("******Finished training, test acc {}******".format(test_accs[-1]))
    print("Finished training!")
    print('Best acc:')
    print(best_acc)

    print('Test acc:')
    print(test_accs)


def validate(valloader, model, criterion, epoch, mode):
    model.eval()
    with torch.no_grad():
        loss_total = 0
        total_sample = 0
        acc_total = 0
        correct = 0
        f1_pred_lbl = []
        f1_true_lbl = []

        for batch_idx, (inputs, targets, length) in enumerate(valloader):
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)


            outputs = model(inputs)
            loss = criterion(outputs, targets)

            _, predicted = torch.max(outputs.data, 1)

            correct += (np.array(predicted.cpu()) ==
                        np.array(targets.cpu())).sum()
            loss_total += loss.item() * inputs.shape[0]
            total_sample += inputs.shape[0]

            f1_pred_lbl.extend(np.array(predicted.cpu()).tolist())
            f1_true_lbl.extend(np.array(targets.cpu()).tolist())

        f1 = f1_score(f1_true_lbl, f1_pred_lbl, average=None)
        avg_f1 = np.mean(f1)

        acc_total = correct/total_sample
        loss_total = loss_total/total_sample

    return loss_total, acc_total, avg_f1


def train(labeled_trainloader, model, optimizer, criterion, epoch):
    model.train()

    for batch_idx, (inputs, targets, length) in enumerate(labeled_trainloader):
        inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
        inputs = inputs.reshape(-1, inputs.shape[-1])
        targets = targets.reshape(-1)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        print('epoch {}, step {}, loss {}'.format(
            epoch, batch_idx, loss.item()))
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    main()
