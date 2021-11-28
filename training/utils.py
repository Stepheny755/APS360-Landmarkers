import torch
import torch.nn as nn

from torchsummary import summary

from torch.utils.data.sampler import SubsetRandomSampler
from adamp import AdamP

from .transforms import get_transforms
from .datasets import GLRv2, GLRv2_5, GLRv2_5_preprocessed
from .models.efficientnet.efficient_net import EfficientNet
from .models.swintransformer.swin_transformer import SwinTransformer
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
            'patch_size': 4,  # (int | tuple(int)): Patch size. Default: 4
            'in_chans': 3,  # (int): Number of input image channels. Default: 3
            'num_class': num_classes,  # (int): Number of classes for classification head. Default: 1000
            'embed_dim': 96,  # (int): Patch embedding dimension. Default: 96
            'depths': [2, 2, 6, 2],  # (tuple(int)): Depth of each Swin Transformer layer. (see above)
            'num_heads': [4, 8, 16, 32],  # (tuple(int)): Number of attention heads in different layers.
            'window_size': 7,  # (int): Window size. Default: 7
            'mlp_ratio': 4.0,  # (float): Ratio of mlp hidden dim to embedding dim. Default: 4
            'qkv_bias': True,  # (bool): If True, add a learnable bias to query, key, value. Default: True
            'qk_scale': None,  # (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
            'drop_rate': 0.0,  # (float): Dropout rate. Default: 0
            'drop_path_rate': 0.1,  # (float): Stochastic depth rate. Default: 0.1
            'ape': False,  # (bool): If True, add absolute position embedding to the patch embedding. Default: False
            'patch_norm': True,  # (bool): If True, add normalization after patch embedding. Default: True
            'use_checkpoint': False  # (bool): Whether to use checkpointing to save memory. Default: False
        }
        model = SwinTransformer(feature_extractor=swin_config['feature_extractor'],
                        img_size=swin_config['img_size'],
                        patch_size=swin_config['patch_size'],
                        in_chans=swin_config['in_chans'],
                        num_classes=swin_config['num_class'],
                        embed_dim=swin_config['embed_dim'],
                        depths=swin_config['depths'],
                        num_heads=swin_config['num_heads'],
                        window_size=swin_config['window_size'],
                        mlp_ratio=swin_config['mlp_ratio'],
                        qkv_bias=swin_config['qkv_bias'],
                        qk_scale=swin_config['qk_scale'],
                        drop_rate=swin_config['drop_rate'],
                        drop_path_rate=swin_config['drop_path_rate'],
                        ape=swin_config['ape'],
                        patch_norm=swin_config['patch_norm'],
                        use_checkpoint=swin_config['use_checkpoint'])
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