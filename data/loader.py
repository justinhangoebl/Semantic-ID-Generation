import torch
import os
from data.amazon_data import AmazonReviews
from torchvision import datasets, transforms
from sklearn.preprocessing import normalize
import pandas as pd
import numpy as np

def load_amazon(category='beauty', normalize_data=True, train=True):
    path = fr"dataset/amazon/processed/data_{category}.pt"
    
    if(not os.path.exists(path)):
        AmazonReviews("dataset/amazon", split=category)
    
    data, _, _ = torch.load(path, weights_only=False)
    
    if normalize_data:
        data['item']['x'] = normalize(data['item']['x'].clone())
        
    data_clean = data['item']['x'][data['item']['is_train']== train]

    return data_clean

def load_mnist(normalize_data=True, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) if normalize_data else transforms.Lambda(lambda x: x),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    dataset = datasets.MNIST(root='./dataset', train=train, download=True, transform=transform)
    
    # Apply transforms and extract data
    data = torch.stack([dataset[i][0] for i in range(len(dataset))])
    
    if not train:
        labels = torch.tensor([dataset[i][1] for i in range(len(dataset))])
        return data, labels

    return data

def load_movie_lens(file_path='./ml-1m/ml-1m.inter', normalize_data=True, train=True):

    # Read the interaction file
    df = pd.read_csv(file_path, sep='\t')
    df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    
    # Convert to numeric and drop invalid rows
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna()
    
    # Create user and item mappings
    unique_users = sorted(df['user_id'].unique())
    unique_items = sorted(df['item_id'].unique())
    
    user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
    item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
    
    # Create interaction matrix
    n_users = len(unique_users)
    n_items = len(unique_items)
    interaction_matrix = np.zeros((n_users, n_items), dtype=np.float32)
    
    # Fill the interaction matrix
    for _, row in df.iterrows():
        user_idx = user_to_idx[row['user_id']]
        item_idx = item_to_idx[row['item_id']]
        interaction_matrix[user_idx, item_idx] = row['rating']
    
    # Normalize if requested
    if normalize_data:
        # Normalize ratings to [0, 1] range
        min_rating = interaction_matrix[interaction_matrix > 0].min()
        max_rating = interaction_matrix[interaction_matrix > 0].max()
        
        # Only normalize non-zero entries
        mask = interaction_matrix > 0
        interaction_matrix[mask] = (interaction_matrix[mask] - min_rating) / (max_rating - min_rating)
    
    # Convert to tensor
    data = torch.tensor(interaction_matrix)
    
    # Simple train/test split (80/20 by users)
    if train:
        split_idx = int(0.8 * n_users)
        data = data[:split_idx]
    else:
        split_idx = int(0.8 * n_users)
        data = data[split_idx:]
        user_ids = torch.tensor(list(range(split_idx, n_users)))
        item_ids = torch.tensor(list(range(n_items)))
        return data, user_ids, item_ids
    
    return data