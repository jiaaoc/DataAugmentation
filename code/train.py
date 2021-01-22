import argparse
import os
import random
import math

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from transformers import *
from sklearn.metrics import f1_score


# from pytorch_transformers import *
from code.normal_bert import ClassificationBert
from torch.autograd import Variable
from torch.utils.data import Dataset
import json
from code.read_data import *


# logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description='PyTorch Data Augmentation')


parser.add_argument('--gpu', default='0,1,2,3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')


# Training parameters
parser.add_argument('--model', type=str, default='bert-base-uncased',
                    help='pretrained model')

parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=4, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--batch-size-u', default=24, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--val-iteration', type=int, default=200,
                    help='frequency of evaluation')
parser.add_argument('--test-batch-size', default=24, type=int, metavar='N',
                    help='train batchsize')

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

parser.add_argument('--n-labeled', type=int, default=20,
                    help='number of labeled data')

parser.add_argument('--un-labeled', default=20000, type=int,
                    help='number of unlabeled data')


# Augmentations
parser.add_argument('--train-aug', default=False, type=bool, metavar='N',
                    help='augment labeled training data')

parser.add_argument('--transform-type', type=str, default='BackTranslation',
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

    # logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    #                     datefmt='%m/%d/%Y %H:%M:%S',
    #                     level=logging.INFO)
    # logger.warning("Device: %s, n_gpu: %s", device, args.n_gpu)


    global best_acc
    # Read dataset and build dataloaders
    train_labeled_set, train_unlabeled_set, val_set, test_set, n_labels = get_data(
        args.data_path, args.n_labeled, args.un_labeled, model=args.model, train_aug=args.train_aug,
        transform_type=args.transform_type, transform_times = args.transform_times)

    # train_labeled_set, train_unlabeled_set, val_set, test_set, n_labels = get_data(
    #     args.data_path, args.n_labeled, args.un_labeled, model=args.model, train_aug=args.train_aug,
    #     transform_type=args.transform_type, transform_times = args.transform_times)

    labeled_trainloader = Data.DataLoader(
        dataset=train_labeled_set, batch_size=args.batch_size, shuffle=True)
    unlabeled_trainloader = Data.DataLoader(
        dataset=train_unlabeled_set, batch_size=args.batch_size_u, shuffle=True)
    val_loader = Data.DataLoader(
        dataset=val_set, batch_size=args.test_batch_size, shuffle=False)
    test_loader = Data.DataLoader(
        dataset=test_set, batch_size=args.test_batch_size, shuffle=False)

    # Define the model, set the optimizer
    model = ClassificationBert(n_labels).cuda()

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    
    optimizer = AdamW(
        [
            {"params": model.bert.parameters(), "lr": args.lrmain},
            {"params": model.linear.parameters(), "lr": args.lrlast},
        ])

    train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss()

    scheduler = None

    test_accs = []

    # logger.info("***** Running training *****")
    # logger.info("  Num labeled examples = %d", len(labeled_trainloader))
    # logger.info("  Num unlabeled examples = %d", len(unlabeled_trainloader))
    # logger.info("  Num Epochs = %d", args.epochs)
    # logger.info("  LAM_u = %s" % str(args.lambda_u))
    # logger.info("  Batch size = %d" % args.batch_size)
    # logger.info("  Max seq length = %d" % 256)

    underscore_data_path = os.path.split(os.path.split(args.data_path)[0])[1].replace("/", "_")
    file_name = "_".join([underscore_data_path, str(args.n_labeled), str(args.un_labeled), str(args.transform_type)])

    data_directory = os.path.join("exp_out", "ssl_" + underscore_data_path)

    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    file = os.path.join(data_directory, file_name)

    f =  open(file, 'w+')


    # Start training
    for epoch in range(args.epochs):

        train(labeled_trainloader, unlabeled_trainloader, model, optimizer,
              scheduler, train_criterion, epoch, n_labels, args.train_aug)

        val_loss, val_acc, val_f1 = validate(
            val_loader, model, criterion, epoch, mode='Valid Stats')


        logger.info("******Epoch {}, val acc {}, val loss {}******".format(epoch,val_acc, val_loss))

        print("epoch {}, val acc {}, val_loss {}".format(
            epoch, val_acc, val_loss))

        f.write(json.dumps({"epoch": epoch, "val_acc": val_acc, "val_f1": val_f1}) + '\n')


        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), file + ".pt")
            # test_loss, test_acc, test_f1 = validate(
            #     test_loader, model, criterion, epoch, mode='Test Stats ')
            # test_accs.append(test_acc)
            # logger.info("******Epoch {}, test acc {}, test loss {}******".format(epoch, test_acc, test_loss))
            # print("epoch {}, test acc {},test loss {}".format(
            #     epoch, test_acc, test_loss))

    model.load_state_dict(torch.load(file + ".pt"))
    test_loss, test_acc, test_f1 = validate(
        test_loader, model, criterion, epoch, mode='Test Stats ')
    f.write(json.dumps({"epoch": epoch, "best_test_acc": test_acc, "best_test_f1": test_f1}) + '\n')
    test_accs.append(test_acc)

    #     print('Epoch: ', epoch)
    #
    #     print('Best acc:')
    #     print(best_acc)
    #
    #     print('Test acc:')
    #     print(test_accs)
    #
    # logger.info("******Finished training, test acc {}******".format(test_accs[-1]))
    # print("Finished training!")
    # print('Best acc:')
    # print(best_acc)
    #
    # print('Test acc:')
    # print(test_accs)


def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, scheduler, criterion, epoch, n_labels, train_aug=False):
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    model.train()

    global total_steps
    global flag
    if flag == 0 and total_steps > args.temp_change:
        print('Change T!')
        args.T = 0.9
        flag = 1

    for batch_idx in range(args.val_iteration):

        total_steps += 1

        if not train_aug:
            try:
                inputs_x, targets_x, inputs_x_length = labeled_train_iter.next()
            except:
                labeled_train_iter = iter(labeled_trainloader)
                inputs_x, targets_x, inputs_x_length = labeled_train_iter.next()
        else:
            try:
                (inputs_x, inputs_x_aug), (targets_x, _), (inputs_x_length,
                                                           inputs_x_length_aug) = labeled_train_iter.next()
            except:
                labeled_train_iter = iter(labeled_trainloader)
                (inputs_x, inputs_x_aug), (targets_x, _), (inputs_x_length,
                                                           inputs_x_length_aug) = labeled_train_iter.next()
        try:
            (inputs_u, inputs_u2,  inputs_ori), (length_u,
                                                 length_u2,  length_ori) = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)

            (inputs_u, inputs_u2, inputs_ori), (length_u, length_u2, length_ori) = unlabeled_train_iter.next()


        batch_size = inputs_x.size(0)
        batch_size_2 = inputs_ori.size(0)
        inputs_x = torch.tensor(inputs_x[:,0])
        targets_x = torch.tensor(targets_x[:,0])

        targets_x = torch.zeros(batch_size, n_labels).scatter_(
            1, targets_x.view(-1, 1), 1)


        inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)

        inputs_u = inputs_u.cuda()
        inputs_u2 = inputs_u2.cuda()
        inputs_ori = inputs_ori.cuda()

        mask = []

        with torch.no_grad():
            # Predict labels for unlabeled data.
            outputs_u = model(inputs_u)
            outputs_u2 = model(inputs_u2)
            outputs_ori = model(inputs_ori)

            # Based on translation qualities, choose different weights here.
            # For AG News: German: 1, Russian: 0, ori: 1
            # For DBPedia: German: 1, Russian: 1, ori: 1
            # For IMDB: German: 0, Russian: 0, ori: 1
            # For Yahoo Answers: German: 1, Russian: 0, ori: 1 / German: 0, Russian: 0, ori: 1
            p = (0 * torch.softmax(outputs_u, dim=1) + 0 * torch.softmax(outputs_u2,
                                                                         dim=1) + 1 * torch.softmax(outputs_ori, dim=1)) / (1)
            # Do a sharpen here.
            pt = p**(1/args.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        mixed = 1
        mix_ = 0

        # if args.co:
        #     mix_ = np.random.choice([0, 1], 1)[0]
        # else:
        #     mix_ = 1
        #
        if mix_ == 1:
            l = np.random.beta(args.alpha, args.alpha)
            if args.separate_mix:
                l = l
            else:
                l = max(l, 1-l)
        else:
            l = 1

        args.mix_layers_set = [1]

        mix_layer = np.random.choice(args.mix_layers_set, 1)[0]
        mix_layer = mix_layer - 1

        if not train_aug:
            all_inputs = torch.cat(
                [inputs_x, inputs_u, inputs_u2, inputs_ori, inputs_ori], dim=0)

            # all_lengths = torch.cat(
            #     [inputs_x_length, length_u, length_u2, length_ori, length_ori], dim=0)

            all_targets = torch.cat(
                [targets_x, targets_u, targets_u, targets_u, targets_u], dim=0)

        else:
            all_inputs = torch.cat(
                [inputs_x, inputs_x_aug, inputs_u, inputs_u2, inputs_ori], dim=0)
            # all_lengths = torch.cat(
            #     [inputs_x_length, inputs_x_length, length_u, length_u2, length_ori], dim=0)
            all_targets = torch.cat(
                [targets_x, targets_x, targets_u, targets_u, targets_u], dim=0)



        args.separate_mix = True

        if args.separate_mix:
            idx1 = torch.randperm(batch_size)
            idx2 = torch.randperm(all_inputs.size(0) - batch_size) + batch_size
            idx = torch.cat([idx1, idx2], dim=0)

        else:
            idx1 = torch.randperm(all_inputs.size(0) - batch_size_2)
            idx2 = torch.arange(batch_size_2) + \
                all_inputs.size(0) - batch_size_2
            idx = torch.cat([idx1, idx2], dim=0)

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        # length_a, length_b = all_lengths, all_lengths[idx]

        args.mix_method = 0

        if args.mix_method == 0:
            # Mix sentences' hidden representations
            # logits = model(input_a, input_b, l, mix_layer)
            # mixed_target = l * target_a + (1 - l) * target_b
            mixed_target = target_a
            logits = model(input_a)

        # elif args.mix_method == 1:
        #     # Concat snippet of two training sentences, the snippets are selected based on l
        #     # For example: "I lova you so much" and "He likes NLP" could be mixed as "He likes NLP so much".
        #     # The corresponding labels are mixed with coefficient as well
        #     mixed_input = []
        #     if l != 1:
        #         for i in range(input_a.size(0)):
        #             length1 = math.floor(int(length_a[i]) * l)
        #             idx1 = torch.randperm(int(length_a[i]) - length1 + 1)[0]
        #             length2 = math.ceil(int(length_b[i]) * (1-l))
        #             if length1 + length2 > 256:
        #                 length2 = 256-length1 - 1
        #             idx2 = torch.randperm(int(length_b[i]) - length2 + 1)[0]
        #             try:
        #                 mixed_input.append(
        #                     torch.cat((input_a[i][idx1: idx1 + length1], torch.tensor([102]).cuda(), input_b[i][idx2:idx2 + length2], torch.tensor([0]*(256-1-length1-length2)).cuda()), dim=0).unsqueeze(0))
        #             except:
        #                 print(256 - 1 - length1 - length2,
        #                       idx2, length2, idx1, length1)
        #
        #         mixed_input = torch.cat(mixed_input, dim=0)
        #
        #     else:
        #         mixed_input = input_a
        #
        #     logits = model(mixed_input)
        #     mixed_target = l * target_a + (1 - l) * target_b
        #
        # elif args.mix_method == 2:
        #     # Concat two training sentences
        #     # The corresponding labels are averaged
        #     if l == 1:
        #         mixed_input = []
        #         for i in range(input_a.size(0)):
        #             mixed_input.append(
        #                 torch.cat((input_a[i][:length_a[i]], torch.tensor([102]).cuda(), input_b[i][:length_b[i]], torch.tensor([0]*(512-1-int(length_a[i])-int(length_b[i]))).cuda()), dim=0).unsqueeze(0))
        #
        #         mixed_input = torch.cat(mixed_input, dim=0)
        #         logits = model(mixed_input, sent_size=512)
        #
        #         #mixed_target = torch.clamp(target_a + target_b, max = 1)
        #         mixed = 0
        #         mixed_target = (target_a + target_b)/2
        #     else:
        #         mixed_input = input_a
        #         mixed_target = target_a
        #         logits = model(mixed_input, sent_size=256)
        #         mixed = 1

        Lx, Lu, w = criterion(logits[:batch_size], mixed_target[:batch_size], logits[batch_size:-batch_size_2],
                                       mixed_target[batch_size:-batch_size_2], logits[-batch_size_2:], epoch+batch_idx/args.val_iteration, mixed)


        # loss = Lx
        loss = Lx + w * Lu

        #max_grad_norm = 1.0
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        # if batch_idx % 1000 == 0:
        # print("epoch {}, step {}, loss {}, Lx {}, Lu {}, Lu2 {}".format(
        #         epoch, batch_idx, loss.item(), Lx.item(), Lu.item(), Lu2.item()))
        print("epoch {}, step {}, loss {}, Lx {}, Lu {}".format(
                epoch, batch_idx, loss.item(), Lx.item(), Lu.item()))


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


            f1_pred_lbl.extend(np.array(predicted.cpu()).tolist())
            f1_true_lbl.extend(np.array(targets.cpu()).tolist())


            correct += (np.array(predicted.cpu()) ==
                        np.array(targets.cpu())).sum()
            loss_total += loss.item() * inputs.shape[0]
            total_sample += inputs.shape[0]

        f1 = f1_score(f1_true_lbl, f1_pred_lbl, average=None)
        avg_f1 = np.mean(f1)

        acc_total = correct/total_sample
        loss_total = loss_total/total_sample

    return loss_total, acc_total, avg_f1


def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class SemiLoss(object):
    # args.margin = 0
    # args.lambda_u_hinge = 0

    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, outputs_u_2, epoch, mixed=1):

        if args.mix_method == 0 or args.mix_method == 1:

            Lx = - \
                torch.mean(torch.sum(F.log_softmax(
                    outputs_x, dim=1) * targets_x, dim=1))

            probs_u = torch.softmax(outputs_u, dim=1)

            Lu = F.kl_div(probs_u.log(), targets_u, None, None, 'batchmean')

            # Lu2 = torch.mean(torch.clamp(torch.sum(-F.softmax(outputs_u, dim=1)
            #                                        * F.log_softmax(outputs_u, dim=1), dim=1) - args.margin, min=0))

        # elif args.mix_method == 2:
        #     if mixed == 0:
        #         Lx = - \
        #             torch.mean(torch.sum(F.logsigmoid(
        #                 outputs_x) * targets_x, dim=1))
        #
        #         probs_u = torch.softmax(outputs_u, dim=1)
        #
        #         Lu = F.kl_div(probs_u.log(), targets_u,
        #                       None, None, 'batchmean')
        #
        #         Lu2 = torch.mean(torch.clamp(args.margin - torch.sum(
        #             F.softmax(outputs_u_2, dim=1) * F.softmax(outputs_u_2, dim=1), dim=1), min=0))
        #     else:
        #         Lx = - \
        #             torch.mean(torch.sum(F.log_softmax(
        #                 outputs_x, dim=1) * targets_x, dim=1))
        #
        #         probs_u = torch.softmax(outputs_u, dim=1)
        #         Lu = F.kl_div(probs_u.log(), targets_u,
        #                       None, None, 'batchmean')
        #
        #         Lu2 = torch.mean(torch.clamp(args.margin - torch.sum(
        #             F.softmax(outputs_u, dim=1) * F.softmax(outputs_u, dim=1), dim=1), min=0))

        return Lx, Lu, args.lambda_u * linear_rampup(epoch)


if __name__ == '__main__':
    main()
