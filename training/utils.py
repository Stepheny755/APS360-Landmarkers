import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler

from .transforms import get_transforms
from .datasets import GLRv2, GLRv2_5, GLRv2_5_preprocessed
from .models.efficientnet.efficient_net import EfficientNet
from .models.senet.se_resnet import se_resnet50

def set_loader(config):

    train_transform, test_transform = get_transforms(
        dataset = config.dataset,
        color_aug = config.color_augmentation,
        dist_factor = config.distortion_factor
    )

    if "GLRv2" in config.dataset:
        if config.dataset == "GLRv2":
            train_dataset = GLRv2(config.data_folder, transform=train_transform)
            test_dataset = GLRv2(config.data_folder, transform=test_transform)
        elif config.dataset == "GLRv2_5":
            train_dataset = GLRv2_5(config.data_folder, transform=train_transform)
            test_dataset = GLRv2_5(config.data_folder, transform=test_transform)
        elif config.dataset == "GLRv2_5_preprocessed":
            train_dataset = GLRv2_5_preprocessed(config.data_folder, transform=train_transform)
            test_dataset = GLRv2_5_preprocessed(config.data_folder, transform=test_transform)
    
        assert list(train_dataset.test_indices) == list(test_dataset.test_indices)
        #create the SubsetRandomSamplers
        train_sampler = SubsetRandomSampler(train_dataset.train_indices)
        val_sampler = SubsetRandomSampler(test_dataset.val_indices)
        test_sampler = SubsetRandomSampler(test_dataset.test_indices)

        #create DataLoader objects to be used in training
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, 
            shuffle=False, 
            batch_size=config.batch_size, 
            sampler=train_sampler, 
            num_workers=config.num_workers)
        
        val_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, 
            shuffle=False, 
            batch_size=config.batch_size, 
            sampler=val_sampler, 
            num_workers=config.num_workers)
        
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, 
            shuffle=False, 
            batch_size=config.batch_size, 
            sampler=test_sampler, 
            num_workers=config.num_workers)
    else: 
        raise NotImplementedError(f"{config.dataset} not implemented!")
    
    return train_loader, val_loader, test_loader

def set_model(config, train_loader):
    model = None
    if "efficientnet" in config.network:
        model = EfficientNet.from_pretrained(
            config.network, 
            num_classes=len(list(train_loader.dataset.class_to_idx.values()))
        )
        if config.freeze_layers == "True":
            for param in model.parameters():
                param.requires_grad = False
        model._fc = nn.Linear(model._fc.in_features, model._fc.out_features)
    elif config.network == "senet-50":
        model = se_resnet50(
            num_classes=len(list(train_loader.dataset.class_to_idx.values()))
        )
        if config.freeze_layers == "True":
            for param in model.parameters():
                param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, model.fc.out_features)
    elif config.network == "swin":
        model = None
    elif config.network == "DeLF+SVM":
        model = None
    else:
        raise NotImplementedError(
            f"{config.network} not implemented!")
    
    criterion = None
    if config.loss == 'CrossEntropyLoss':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(
            f"{config.loss} not implemented!")

    return model.cuda(), criterion

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
        "scheduler" : scheduler.state_dict() if scheduler else scheduler
    }

    torch.save(save_dict, save_file)