import torch
import os
from data.amazon_data import AmazonReviews
from torchvision import datasets, transforms
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

def load_movielens(normalize_data=True, train=True):
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