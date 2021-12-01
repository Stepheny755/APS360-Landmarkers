import os
import pandas as pd
import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn
from sklearn import svm

from training import (
                      parse_option, 
                      set_loader)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

def main(folder_path):
    config = parse_option()
    print(config)
    # build data loader
    train_loader, val_loader, test_loader = set_loader(config)

    # build model and criterion
    model = models.resnet50(pretrained=True)
    model.fc = Identity()
    model = model.cuda()
    model.train()

    embedded_vectors = []
    labels_np = []
    for idx, (images, labels) in enumerate(train_loader):
        with torch.no_grad():
            images = images.cuda(non_blocking=True)
            embedded_vectors.append(model(images).detach().cpu().numpy())
            labels_np.append(labels.detach().numpy())

    print(np.concatenate(embedded_vectors, axis=0).shape)
    print(np.concatenate(labels_np, axis=0).shape)
    #train svm
    clf = svm.SVC()
    clf.fit(
        np.concatenate(embedded_vectors, axis=0), 
        np.concatenate(labels_np, axis=0))
    
    val_embedded_vectors = []
    val_labels_np = []

    for idx, (images, labels) in enumerate(test_loader):
        with torch.no_grad():
            images = images.cuda(non_blocking=True)
            val_embedded_vectors.append(model(images).detach().cpu().numpy())
            val_labels_np.append(labels.detach().numpy())
    
    preds = clf.predict(np.concatenate(val_embedded_vectors, axis=0))
    corr = preds == np.concatenate(val_labels_np, axis=0)
    print(f"percent correct is {np.sum(corr)/corr.shape[0]}")

    

if __name__ == "__main__":
    folder_path = "eval"
    main(folder_path)