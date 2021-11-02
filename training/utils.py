import torch

def set_loader(config):
    pass

def set_model(config, train_loader):
    model = None
    if config.network == "efficientnet":
        model = None
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
    pass