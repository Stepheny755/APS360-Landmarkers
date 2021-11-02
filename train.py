from training import train
from training import parse_option
from torch.utils.tensorboard import SummaryWriter


def main():
    config = parse_option()

    # build data loader
    train_loader, val_loader, test_loader = set_loader(config)

    # build model and criterion
    model, criterion = set_model(config, train_loader)

    # build optimizer and scheduler
    optimizer = set_optimizer(config, model)
    scheduler = set_scheduler(config, optimizer)

    # Metrics Calculation
    performance_statistics = {}
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
    writer = SummaryWriter(f'{os.path.join(config.eval_folder, losses_filename)}')


    """___________________Training____________________"""
    for epoch in range(1, config.epochs + 1):

        # train and test for one epoch
        train_loss, test_loss, train_acc, test_acc = one_epoch_iteration(train_loader, test_loader, model, criterion,
                                                                         optimizer, epoch, config, history)
        scheduler.step()

        losses["train_loss"].append(train_loss)
        losses["test_loss"].append(test_loss)
        losses["train_acc"].append(train_acc)
        losses["test_acc"].append(test_acc)

        losses['learning_rate'].append(optimizer.param_groups[0]['lr'])

        # Performance metrics
        if train_acc:
            performance_statistics[f'train_acc{epoch}'] = train_acc
        if test_acc:
            performance_statistics[f'test_acc_{epoch}'] = test_acc


        # save the model
        if epoch % config.save_freq == 0:
            save_file = os.path.join(
                config.save_folder, 'checkpoints_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, config, epoch, save_file)

    # Store the losses in a dataframe
    loss_df = pd.DataFrame(data=losses)
    # Save the loss dataframe in a excel
    loss_df.to_excel(os.path.join(config.eval_folder, losses_filename))
    # Plot the losses
    plot_loss_df(loss_df, config)

    # save the last model
    save_file = os.path.join(config.save_folder, 'last.pth')
    save_model(model, optimizer, config, config.epochs, save_file)

if __name__ == '__main__':
    main()
