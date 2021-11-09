from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from training import (one_epoch_iteration,
                      parse_option, 
                      set_loader, 
                      set_model, 
                      set_optimizer, 
                      set_scheduler,
                      save_model)

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img * np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1) + np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)     # unnormalize
    npimg = img.numpy().clip(0, 1)
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def main():
    config = parse_option()
    # build data loader
    train_loader, val_loader, test_loader = set_loader(config)

    # build model and criterion
    model, criterion = set_model(config, test_loader)
    # get some random training images
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    # create grid of images
    img_grid = torchvision.utils.make_grid(images)

    # show images
    matplotlib_imshow(img_grid, one_channel=False)

    # write to tensorboard
    writer = SummaryWriter("./runs")
    writer.add_image('dsafdsafdsa', img_grid)
    images = images.cuda()
    
    writer.add_graph(model, images)
    writer.close()

if __name__ == '__main__':
    main()