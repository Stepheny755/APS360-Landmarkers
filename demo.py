import os
import numpy as np
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class UofTData(ImageFolder):
    train_folder_name = "uoft_Dataset"
    test_folder_name = "uoft_Dataset_test"


    def __init__(self, root, transform=None, train=True):
        if train:
            super(UofTData, self).__init__(root=os.path.join(root, self.train_folder_name), transform=transform)
        else:
            super(UofTData, self).__init__(root=os.path.join(root, self.test_folder_name), transform=transform)

def plot_classes_preds(images, preds, labels, classes, k=5):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(len(labels)):
        ax = fig.add_subplot(len(labels), k, idx+1, xticks=[], yticks=[])
        img = mpimg.imread(images[idx])
        plt.imshow(img)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig

def plot_imgs(images, m=3, k=3):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(m*k):
        ax = fig.add_subplot(m*k, k, idx+1, xticks=[], yticks=[])
        img = mpimg.imread(images[idx])
        plt.imshow(img)
    return fig