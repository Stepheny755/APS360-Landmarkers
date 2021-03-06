from datetime import datetime
import collections
import os

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from training import (one_epoch_iteration,
                      parse_option, 
                      set_loader, 
                      set_model, 
                      set_optimizer, 
                      set_scheduler,
                      save_model)



def main():
    config = parse_option()
    print(config)
    # build data loader
    train_loader, val_loader, test_loader = set_loader(config)

    # build model and criterion
    model, criterion = set_model(config, train_loader)

    # build optimizer and scheduler
    optimizer = set_optimizer(config, model)
    if config.scheduler != "None":
        scheduler = set_scheduler(config, optimizer)
    else:
        scheduler = None

    # Metrics Calculation
    cur_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    losses = collections.defaultdict(list)
    losses_filename = "losses_" + \
                      f"{config.loss}_" + \
                      f"{cur_time}_" + \
                      f"{config.network}_" + \
                      f"{config.dataset}_" + \
                      f"{config.optimizer}" + \
                      f"_{config.scheduler}" + \
                      f"_{config.step_size}" + \
                      f"_{config.lr_decay_rate}" + \
                      f"_LR={config.learning_rate}" + \
                      ".xlsx".replace(' ', '-')
    writer = SummaryWriter(f'{os.path.join(config.eval_folder, "runs", losses_filename.replace(".xlsx", ""))}')

    best_val_acc = 0
    early_stopping_counter = 0

    """___________________Training____________________"""
    for epoch in range(1, config.epochs + 1):

        # train and test for one epoch
        train_loss, val_loss, train_acc, val_acc = one_epoch_iteration(train_loader, val_loader, model, criterion,
                                                                         optimizer, epoch, config, writer)
        if scheduler:
            scheduler.step()

        losses["train_loss"].append(train_loss)
        losses["val_loss"].append(val_loss)
        losses["train_acc"].append(train_acc)
        losses["val_acc"].append(val_acc)

        losses['learning_rate'].append(optimizer.param_groups[0]['lr'])

        # save the model
        if (epoch-1) % config.save_freq == config.save_freq - 1:
            save_file = os.path.join(
                config.save_folder, 
                f'{cur_time}_checkpoints_epoch_{epoch}.pth')
            save_model(model, optimizer, scheduler, config, epoch, save_file)

        if val_acc > best_val_acc:
            early_stopping_counter = 0
            best_val_acc = val_acc
            save_file = os.path.join(config.save_folder, f'{cur_time}_best.pth')
            save_model(model, optimizer, scheduler, config, epoch, save_file)
        else:
            early_stopping_counter += 1
            if config.early_stopping > 0:
                if early_stopping_counter >= config.early_stopping:
                    print(f"ran for {early_stopping_counter} without increase in val acc")
                    print(f"stopping early...")
                    break

    # save the last model
    save_file = os.path.join(config.save_folder, f'{cur_time}_last.pth')
    save_model(model, optimizer, scheduler, config, config.epochs, save_file)

    # Store the losses in a dataframe
    loss_df = pd.DataFrame(data=losses)
    # Save the loss dataframe in a excel
    loss_df.to_excel(os.path.join(config.eval_folder, losses_filename))

if __name__ == '__main__':
    main()
