import os
import sys

import time
import numpy as np

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import collections

import warnings
warnings.filterwarnings("ignore")

from utils import AverageMeter
from datetime import datetime
from classifier import get_model, accuracyADP



def one_epoch_iteration(train_loader, test_loader, model, criterion,
                        optimizer, epoch, config, writer):

    start_time = time.time()

    """---------------Training------------------"""
    train_loss, train_acc = train(train_loader, model, criterion,
                                  optimizer, epoch, config, writer)

    """-----------------Testing------------------"""
    test_loss, test_acc = test(test_loader, model, criterion, config, writer)
    end_time = time.time()

    print('epoch {}\t'
          'Train_loss {:.3f}\t'
          'Train_acc {} \t'
          'Test_loss {:.3f} \t'
          'Test_acc {} \t'
          'total_time {:.2f}'.format(epoch,
                                     train_loss[2] if isinstance(train_loss, list) else train_loss,
                                     train_acc,
                                     test_loss[2] if isinstance(test_loss, list) else test_loss,
                                     test_acc,
                                     end_time - start_time))
    sys.stdout.flush()

    return train_loss, test_loss, train_acc, test_acc

def train(train_loader, model, criterion, optimizer, epoch, config, param_history):
    """one epoch training"""
    model.train()

    losses = AverageMeter()
    train_acc = None
    if config.loss == "CrossEntropyLoss":
        train_acc = AverageMeter()
        

    """___________________Training____________________"""

    for idx, (images, labels) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        batch_size = labels.shape[0]

        # compute loss

        if config.loss == 'MCLoss':
            features, gaussian_params = model(images)
            NLLLoss, GJSDLoss, train_loss = criterion(inp=features,
                                                      pi=gaussian_params['pi'],
                                                      mean=gaussian_params['mean'],
                                                      var=gaussian_params['var'],
                                                      labels=labels,
                                                      var_weights=model.gmm.var_weight)
        elif config.loss == "MultiLabelSoftMarginLoss":
            outputs = model(images)
            train_loss = criterion(outputs, labels)
            if config.dataset == "ADP-Release1":
                preds = (F.sigmoid(torch.tensor(outputs)) > 0.5).int()

                acc1, acc5 = accuracyADP(preds, labels)
                train_acc.update(acc1.double() / (labels.shape[0] * labels.shape[1]), images.shape[0])
            else:
                raise NotImplementedError("Only ADP supported right now")

        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(config.loss))

        # update metric
        losses.update(train_loss.item(), batch_size)

        if config.loss == "MCLoss":
            NLLLosses.update(NLLLoss.item(), batch_size)
            GJSDLosses.update(GJSDLoss.item(), batch_size)

        # SGD with momentum
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if not param_history == None and not config.fine_tune:
            param_history.update(pi=gaussian_params['pi'].clone().detach().cpu(),
                                 mu=gaussian_params['mean'].clone().detach().cpu(),
                                 sigma=gaussian_params['var'].clone().detach().cpu())

        # print info
        if (idx + 1) % config.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                        epoch, idx + 1, len(train_loader),  loss=losses))
            sys.stdout.flush()
    if config.loss == "MCLoss":
        return list([NLLLosses.avg, GJSDLosses.avg, losses.avg]), train_acc
    elif config.loss == "MultiLabelSoftMarginLoss":
        return float(losses.avg), train_acc.avg.item()
    else:
        return losses, train_acc