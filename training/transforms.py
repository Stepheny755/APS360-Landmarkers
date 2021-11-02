import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.utils.data 
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler

class ColorDistortion:
    def __init__(self, distortion):
        self.distortion = distortion

    def __call__(self, image):
        color_jitter = transforms.ColorJitter(0.8 * self.distortion, 0.8 * self.distortion,
                                              0.8 * self.distortion, 0.2 * self.distortion)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=1.0)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([
            rnd_color_jitter,
            # rnd_gray
        ])
        transformed_image = color_distort(image)
        return transformed_image

def get_transforms(config):

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),     #resizes so that shorter edge to 224 pixels
        transforms.CenterCrop(224),
        transforms.Normalize( #TODO CHANGE TO MEAN AND STD OF SMALL DATASET
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomRotation(-180),
        transforms.RandomAffine(-180)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),     #resizes so that shorter edge to 224 pixels
        transforms.CenterCrop(224),
        transforms.Normalize( #TODO CHANGE TO MEAN AND STD OF SMALL DATASET
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return train_transform, test_transform