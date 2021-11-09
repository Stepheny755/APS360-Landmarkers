import os
import sys

import time
import warnings
warnings.filterwarnings("ignore")

import torch


def one_epoch_iteration(train_loader, val_loader, model, criterion,
                        optimizer, epoch, config, writer):

    start_time = time.time()

    """---------------Training------------------"""
    train_loss, train_acc = train(train_loader, model, criterion,
                                  optimizer, epoch, config, writer)

    """-----------------Testing------------------"""
    val_loss, val_acc = validate(val_loader, model, criterion, epoch,
                                   config, writer)
    end_time = time.time()

    print('epoch {}\t'
          'Train_loss {:.3f}\t'
          'Train_acc {} \t'
          'Val_loss {:.3f} \t'
          'Val_acc {} \t'
          'total_time {:.2f}'.format(epoch,
                                     train_loss,
                                     train_acc,
                                     val_loss,
                                     val_acc,
                                     end_time - start_time))
    sys.stdout.flush()

    return train_loss, val_loss, train_acc, val_acc

def train(train_loader, model, criterion, optimizer, epoch, config, writer):
    """one epoch training"""
    model.train()

    losses = 0
    train_acc = 0
        
    """___________________Training____________________"""

    for idx, (images, labels) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        batch_size = labels.shape[0]

        # compute loss

        if config.loss == "CrossEntropyLoss":
            outputs = model(images)
            loss = criterion(outputs, labels)
            if "GLRv2" in config.dataset:
                preds = torch.argmax(outputs, dim=1)
                acc = torch.sum(preds == labels) / batch_size
                train_acc += acc
            else:
                raise NotImplementedError("Only GLRv2 supported right now")

        else:
            raise ValueError('Loss method not supported: {}'.
                             format(config.loss))

        # update metric
        losses += loss.item()

        # optimize network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print info
        if (idx + 1) % config.print_freq == config.print_freq - 1:
            writer.add_scalar('training loss',
                              losses / (config.print_freq * ((idx + 1) // config.print_freq + 1)),
                              (epoch - 1) * len(train_loader) + idx)
            writer.add_scalar('training acc1',
                              train_acc / (config.print_freq * ((idx + 1) // config.print_freq + 1)),
                              (epoch - 1) * len(train_loader) + idx)

    return losses/(idx+1), train_acc/(idx+1)

def validate(val_loader, model, criterion, epoch, config, writer):
    """one epoch evaluation"""
    model.eval()

    losses = 0
    val_acc = 0
        
    """___________________Evaluating____________________"""
    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            batch_size = labels.shape[0]

            # compute loss

            if config.loss == "CrossEntropyLoss":
                outputs = model(images)
                loss = criterion(outputs, labels)
                if config.dataset == "GLRv2":
                    preds = torch.argmax(outputs, dim=1)
                    acc = torch.sum(preds == labels) / batch_size
                    val_acc += acc
                else:
                    raise NotImplementedError("Only GLRv2 supported right now")

            else:
                raise ValueError('Loss method not supported: {}'.
                                format(config.loss))

            # update metric
            losses += loss.item()

            # print info
            if (idx + 1) % config.print_freq == config.print_freq - 1:
                writer.add_scalar('validation loss',
                                losses / (config.print_freq * ((idx + 1) // config.print_freq + 1)),
                                (epoch - 1) * len(val_loader) + idx)
                writer.add_scalar('validation acc1',
                                val_acc / (config.print_freq * ((idx + 1) // config.print_freq + 1)),
                                (epoch - 1) * len(val_loader) + idx)

    return losses/(idx+1), val_acc/(idx+1)