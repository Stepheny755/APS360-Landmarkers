from numpy.lib.npyio import save
import torch
from .models.efficientnet.efficient_net import EfficientNet

def set_loader(config):
    pass

def set_model(config, train_loader):
    model = None
    if config.network == "efficientnet-B3":
        model = EfficientNet.from_pretrained(config.network, 
        num_classes=len(list(train_loader.class_to_idx.values())))
    elif config.network == "senet":
        model = None
    elif config.network == "swin":
        model = None
    elif config.network == "DeLF+SVM":
        model = None
    else:
        raise NotImplementedError(
            f"{config.network} not implemented!")
    return model

def set_optimizer(config, model):
    optimizer = None
    if config.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay)
    elif config.optimizer == "AdamP":
        optimizer = torch.optim.AdamP(
            model.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay)
    elif config.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay)
    else:
        raise NotImplementedError(
            f"{config.optimizer} not implemented!")

    return optimizer

def set_scheduler(config, optimizer):
    scheduler = None
    if config.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=config.step_size,
            gamma=config.lr_decay_rate)
    else:
        raise NotImplementedError(
            f"{config.scheduler} not implemented!")
    
    return scheduler

def save_model(model, optimizer, scheduler, config, epoch, save_file):
    save_dict = {
        "model_state_dict" : model.state_dict(),
        "optimizer_state_dict" : optimizer.state_dict(),
        "epoch" : epoch,
        "config" : config,
        "scheduler" : scheduler.state_dict()
    }

    torch.save(save_dict, save_file)