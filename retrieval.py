import torch.nn.functional as F

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

