import torch.nn.functional as F

def get_knn(query, models, train_loader, k=3):
    #TODO: ADD GPU SUPPORT TO THIS 
    """
    Takes in a query vector, feeds it into the trained models, and retrieves its k-nearest neighbours
    measured by cosine similarity from EACH of the three models. The nearest neighbours are obtained 
    from the training dataset without any extra augmented points.

    Input Arguments:

    query: a query image that has been preprocessed and can be fed into the models
        must have dimensions 1 x 3 x 224 x 224 (unsqueeze if needed)
    models: a list/iterable of the trained models, where
        models[0]: EfficientNet
        models[1]: SENet
        models[2]: swin transformer
        models[3]: DeLF + SVM
    train_loader: the DataLoader object with all the training data in it
    k: the number of nearest neighbours (duh)

    Output Arguments:

    eff_net_matches: list of top k matches from EfficientNet
    se_net_matches: list of top k matches from SENEt
    swin_matches: list of top k matches from Swin Transformer
    delf_matches: list of top k matches from DeLF + SVM
    """

    #need to iterate through training set and store embeddings of each training example 
    #if already done, remove this
    effnet_image_list = []
    for idx, (image, label) in enumerate(train_loader):
        batch_size = image.shape[0]
        for i in range(batch_size):
            effnet_image_list.append(models[0](image[i].unsqueeze(0)))


    #EfficientNet
    query_eff_net = models[0](query) #embedding of query from EfficientNet
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

