import torch
import os
from data.amazon_data import AmazonReviews
from torch.utils.data import DataLoader
from sklearn.preprocessing import normalize

def load_amazon(category='beauty', batch_size=32, normalize_data=True, train=True):
    path = fr"dataset/amazon/processed/data_{category}.pt"
    
    if(not os.path.exists(path)):
        AmazonReviews("dataset/amazon", split=category)
    
    data, _, _ = torch.load(path, weights_only=False)
    
    if normalize_data:
        data['item']['x'] = normalize(data['item']['x'].clone())
        
    data_clean = data['item']['x'][data['item']['is_train']== train]

    train_loader = DataLoader(data_clean, batch_size=batch_size, shuffle=True)
    
    return train_loader, data_clean