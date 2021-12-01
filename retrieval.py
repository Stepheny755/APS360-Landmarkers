from re import L
import torch.nn.functional as F
import numpy as np
import torch
import pickle
import os
import torch.nn as nn

class KNNClassifier():
    def __init__(self, models_list, k=5) -> None:
        self.models = models_list
        self.embeddings = None
        self.fns = None
        self.k = k

    def load_embeddings(self, emb_fl, label_fl, fn_fl):
        self.embeddings = [np.load(f) for f in emb_fl]
        self.labels = [np.load(f) for f in label_fl]
        self.fns = [np.load(f) for f in fn_fl]
    
    def add_embeddings(self, loader, name_loader):
        with torch.no_grad():
            for idx, model in enumerate(self.models):
                labels_np = []
                embeddings_np = []
                fns = []
                device = next(model.parameters()).device
                for i, ((images, labels), (fn, fn_label)) in enumerate(zip(loader, name_loader)):
                    images = images.to(device)
                    embedded_imgs = model.embedding(images).detach().cpu().numpy().copy()
                    embeddings_np.append(embedded_imgs)
                    labels_np.append(labels.detach().numpy().copy())
                    fns.append(fn)

                self.embeddings[idx] = np.concatenate([self.embeddings[idx], *embeddings_np], axis=0, dtype=np.float32)
                self.labels[idx] = np.concatenate([self.labels[idx], *labels_np], axis=0, dtype=np.int64)
                self.fns[idx] = np.concatenate([self.fns[idx], *fns], axis=0)

    def create_embeddings(self, loader, name_loader):
        self.embeddings = [[] for model in self.models]
        self.labels = [[] for model in self.models]
        self.fns = [[] for model in self.models]
        with torch.no_grad():
            for idx, model in enumerate(self.models):
                labels_np = []
                embeddings_np = []
                fns = []
                device = next(model.parameters()).device
                for i, ((images, labels), (fn, fn_label)) in enumerate(zip(loader, name_loader)):
                    images = images.to(device)
                    embedded_imgs = model.embedding(images).detach().cpu().numpy().copy()
                    embeddings_np.append(embedded_imgs)
                    labels_np.append(labels.detach().numpy().copy())
                    fns.append(fn)
                self.embeddings[idx] = np.concatenate(embeddings_np, axis=0, dtype=np.float32)
                self.labels[idx] = np.concatenate(labels_np, axis=0, dtype=np.int64)
                self.fns[idx] = np.concatenate(fns, axis=0)

    def save_embeddings(self, root):
        for idx in range(len(self.models)):
            np.save(os.path.join(root, f"model_{idx}_embeddings.npy"), self.embeddings[idx])
            np.save(os.path.join(root, f"model_{idx}_labels.npy"), self.labels[idx])
            np.save(os.path.join(root, f"model_{idx}_fns.npy"), self.fns[idx])

    def predict(self, x):
        pred_embs = []
        cos = nn.CosineSimilarity(dim=1)
        with torch.no_grad():
            pred_embs = [model.embedding(x.to(next(model.parameters()).device)) 
                        for model in self.models]
            

            cos_sims = [cos(pred_emb.unsqueeze(2), torch.tensor(self.embeddings[idx]).to(next(self.models[idx].parameters()).device).T.unsqueeze(0)) 
                        for idx, pred_emb in enumerate(pred_embs)]

            top_indices = [torch.topk(cos_sim, self.k, dim=1, largest=True)[1] for cos_sim in cos_sims]

            # top_embeddings = [self.embeddings[idx][top_index.detach().cpu().numpy()] for idx, top_index in enumerate(top_indices)]

            top_classes = [self.labels[idx][indices.detach().cpu().numpy()] for idx, indices in enumerate(top_indices)]
            top_classes_images = [self.fns[idx][indices.detach().cpu().numpy()] for idx, indices in enumerate(top_indices)]
            top_classes = np.concatenate(top_classes, axis=1, dtype=np.int64)
            top_class = np.array([np.bincount(arr).argmax() for arr in top_classes])

            return top_class, top_classes_images
            

def get_knn_effnet(query, effnet, train_loader, k=3):
    #TODO: ADD GPU SUPPORT TO THIS 
    """
    Takes in a query vector, feeds it into effnet, and retrieves its k-nearest neighbours
    measured by cosine similarity. The nearest neighbours are obtained 
    from the training dataset without any extra augmented points.

    Input Arguments:

    query: a query image that has been preprocessed and can be fed into the models
        must have dimensions 1 x 3 x 224 x 224 (unsqueeze if needed)
    effnet: an EfficientNet model object
    train_loader: the DataLoader object with all the training data in it
    k: the number of nearest neighbours (duh)

    Output Arguments:

    eff_net_matches: list of top k matches from EfficientNet
    """

    #need to iterate through training set and store embeddings of each training example 
    #if already done, remove this
    effnet_image_list = []
    for idx, (image, label) in enumerate(train_loader):
        batch_size = image.shape[0]
        for i in range(batch_size):
            effnet_image_list.append(effnet.embedding(image[i].unsqueeze(0)))


    #EfficientNet
    query_eff_net = effnet.embedding(query) #embedding of query from EfficientNet
    eff_net_matches = []
    eff_net_dist_scores = {}

    for j in range(len(effnet_image_list)):
        if j == 1:
            print(effnet_image_list[j].shape)
        eff_net_dist_scores[j] = float(F.cosine_similarity(query_eff_net, effnet_image_list[j]))
    match_indices = sorted(eff_net_dist_scores, key=eff_net_dist_scores.get, reverse=True)[:k]

    for l in range(k):
        eff_net_matches.append(effnet_image_list[match_indices[l]])

    return eff_net_matches 

def get_knn_resnet(query, resnet, train_loader, k=3):
    #TODO: ADD GPU SUPPORT TO THIS 
    """
    Takes in a query vector, feeds it into resnet, and retrieves its k-nearest neighbours
    measured by cosine similarity. The nearest neighbours are obtained 
    from the training dataset without any extra augmented points.

    Input Arguments:

    query: a query image that has been preprocessed and can be fed into the models
        must have dimensions 1 x 3 x 224 x 224 (unsqueeze if needed)
    resnet: an SE ResidualNet model object
    train_loader: the DataLoader object with all the training data in it
    k: the number of nearest neighbours (duh)

    Output Arguments:

    eff_net_matches: list of top k matches from EfficientNet
    """

    #need to iterate through training set and store embeddings of each training example 
    #if already done, remove this
    resnet_image_list = []
    for idx, (image, label) in enumerate(train_loader):
        batch_size = image.shape[0]
        for i in range(batch_size):
            resnet_image_list.append(resnet.embedding(image[i].unsqueeze(0)))


    #EfficientNet
    query_res_net = resnet.embedding(query) #embedding of query from SE ResNet
    res_net_matches = []
    res_net_dist_scores = {}

    for j in range(len(resnet_image_list)):
        if j == 1:
            print(resnet_image_list[j].shape)
        res_net_dist_scores[j] = float(F.cosine_similarity(query_res_net, resnet_image_list[j]))
    match_indices = sorted(res_net_dist_scores, key=res_net_dist_scores.get, reverse=True)[:k]

    for l in range(k):
        res_net_matches.append(resnet_image_list[match_indices[l]])

    return res_net_matches

