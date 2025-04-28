import torch
import os
from data.amazon_data import AmazonReviews
from sklearn.preprocessing import normalize

def load_amazon(category='beauty', normalize_data=True, train=True):
    path = fr"dataset/amazon/processed/data_{category}.pt"
    
    if(not os.path.exists(path)):
        AmazonReviews("dataset/amazon", split=category)
    
    data, _, _ = torch.load(path, weights_only=False)
    
    if normalize_data:
        data['item']['x'] = normalize(data['item']['x'].clone())
        
    data_clean = data['item']['x'][data['item']['is_train']== train]

    return data_clean