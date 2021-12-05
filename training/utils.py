from numpy import concatenate
import torch
import torch.nn as nn

from torchsummary import summary

from torch.utils.data.sampler import SubsetRandomSampler, Sampler
from adamp import AdamP

from .transforms import get_transforms
from .datasets import GLRv2, GLRv2_5, GLRv2_5_preprocessed
from .models.efficientnet.efficient_net import EfficientNet
from .models.swintransformer.swin_transformer import SwinTransformer
from .models.senet.se_resnet import se_resnet50


class SubsetSampler(Sampler):
    def __init__(self, mask):
        self.mask = mask

    def __iter__(self):
        return (i for i in self.mask)

    def __len__(self):
        return len(self.mask)

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
        elif config.dataset == "GLRv2_5_preprocessed_file_names":
            train_dataset = GLRv2_5_preprocessed(config.data_folder, transform=None)
            test_dataset = GLRv2_5_preprocessed(config.data_folder, transform=None)
        elif config.dataset == "GLRv2_file_names":
            train_dataset = GLRv2(config.data_folder, transform=None)
            test_dataset = GLRv2(config.data_folder, transform=None)
    
        assert list(train_dataset.test_indices) == list(test_dataset.test_indices)

        #create the SubsetRandomSamplers
        if config.random == "True":
            train_sampler = SubsetRandomSampler(train_dataset.train_indices)
            val_sampler = SubsetRandomSampler(test_dataset.val_indices)
            test_sampler = SubsetRandomSampler(test_dataset.test_indices)
        elif config.random == "False":
            train_sampler = SubsetSampler(train_dataset.train_indices)
            val_sampler = SubsetSampler(test_dataset.val_indices)
            test_sampler = SubsetSampler(test_dataset.test_indices)

        if "file_names" in config.dataset:
            train_dataset = train_dataset.imgs
            test_dataset = test_dataset.imgs

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
        if config.from_pretrained:
            model = EfficientNet.from_pretrained(
                config.network, 
                num_classes=len(list(train_loader.dataset.class_to_idx.values()))
            )
        else:
            model = EfficientNet.from_name(
                config.network, 
                override_params={
                    "num_classes": len(list(train_loader.dataset.class_to_idx.values()))
                }
            )
        if config.freeze_layers == "True":
            for param in model.parameters():
                param.requires_grad = False
        model._fc = nn.Linear(model._fc.in_features, model._fc.out_features)
        
    elif config.network == "senet-50":
        model = se_resnet50(
            num_classes=len(list(train_loader.dataset.class_to_idx.values())),
            pretrained=True if config.from_pretrained == "True" else False
        )
        if config.freeze_layers == "True":
            for param in model.parameters():
                param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, model.fc.out_features)

    elif config.network == "swin":
        num_classes = len(list(train_loader.dataset.class_to_idx.values()))
        swin_config = {
            'feature_extractor': False,  # (bool): If True, drop last fc
            'img_size': 224,  # (int | tuple(int)): Input image size. Default 224
            'num_class': num_classes,  # (int): Number of classes for classification head. Default: 1000
            'embed_dim': 96,  # (int): Patch embedding dimension. Default: 96
            'depths': [2, 2, 6, 2],  # (tuple(int)): Depth of each Swin Transformer layer. (see above)
            'num_heads': [4, 8, 16, 32],  # (tuple(int)): Number of attention heads in different layers.
        }
        model = SwinTransformer(feature_extractor=swin_config['feature_extractor'],
                        img_size=swin_config['img_size'],
                        num_classes=swin_config['num_class'],
                        embed_dim=swin_config['embed_dim'],
                        depths=swin_config['depths'],
                        num_heads=swin_config['num_heads'])
        if config.freeze_layers == "True":
            for param in model.parameters():
                # print(param)
                param.requires_grad = False
        model.head = nn.Linear(model.head.in_features, model.head.out_features)

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
        optimizer = AdamP(
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