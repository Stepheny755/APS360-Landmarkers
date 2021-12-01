import os
import numpy as np
from torchvision.datasets import ImageFolder

class GLRv2(ImageFolder):
    folder_name = "landmark-recognition-1k"
    train_ratio = 0.75
    val_ratio = 0.15
    test_ratio = 0.1

    def __init__(self, root, transform=None):
        super(GLRv2, self).__init__(root=os.path.join(root, self.folder_name), transform=transform)
        
        #get indices to later throw into the random sampler
        indices = np.arange(len(self))

        #randomly split the data indices into train, test, and validation splits, with a fixed seed for reproducible results 
        np.random.seed(100)
        np.random.shuffle(indices)

        train_split = round(len(self) * self.train_ratio)
        val_split = train_split + round(len(self) * self.val_ratio)

        #now split up the dataset itself
        self.train_indices = indices[:train_split]
        self.val_indices = indices[train_split:val_split]
        self.test_indices = indices[val_split:]

class GLRv2_5(ImageFolder):
    folder_name = "landmark-recognition-1k-5"
    train_ratio = 0.75
    val_ratio = 0.15
    test_ratio = 0.1

    def __init__(self, root, transform=None):
        super(GLRv2_5, self).__init__(root=os.path.join(root, self.folder_name), transform=transform)
        
        #get indices to later throw into the random sampler
        indices = np.arange(len(self))

        #randomly split the data indices into train, test, and validation splits, with a fixed seed for reproducible results 
        np.random.seed(100)
        np.random.shuffle(indices)

        train_split = round(len(self) * self.train_ratio)
        val_split = train_split + round(len(self) * self.val_ratio)

        #now split up the dataset itself
        self.train_indices = indices[:train_split]
        self.val_indices = indices[train_split:val_split]
        self.test_indices = indices[val_split:]

class GLRv2_5_preprocessed(ImageFolder):
    folder_name = "landmark-recognition-1k-5-preprocessed"
    train_ratio = 0.75
    val_ratio = 0.15
    test_ratio = 0.1

    def __init__(self, root, transform=None):
        super(GLRv2_5_preprocessed, self).__init__(root=os.path.join(root, self.folder_name), transform=transform)
        
        #get indices to later throw into the random sampler
        indices = np.arange(len(self))

        #randomly split the data indices into train, test, and validation splits, with a fixed seed for reproducible results 
        np.random.seed(100)
        np.random.shuffle(indices)

        train_split = round(len(self) * self.train_ratio)
        val_split = train_split + round(len(self) * self.val_ratio)

        #now split up the dataset itself
        self.train_indices = indices[:train_split]
        self.val_indices = indices[train_split:val_split]
        self.test_indices = indices[val_split:]