import numpy as np
import torch

import torchvision.transforms as transforms


class Dataset_Transforms():

    def __init__(self):
        self.center_crop_width = 224
        self.resize_width = 224

    def get_data_transforms(self):
        data_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224)
        ])

        return data_transforms

