import torch
import os
from data.amazon_data import AmazonReviews
from sklearn.preprocessing import normalize
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch

def load_amazon(category='beauty', normalize_data=True, train=True):
    path = fr"dataset/amazon/processed/data_{category}.pt"
    
    if(not os.path.exists(path)):
        AmazonReviews("dataset/amazon", split=category)
    
    data, _, _ = torch.load(path, weights_only=False)
    
    if normalize_data:
        data['item']['x'] = normalize(data['item']['x'].clone())
        
    data_clean = data['item']['x'][data['item']['is_train']== train]

    return data_clean

def load_movie_lens(category='100k', dimension="user", train=True, raw=True):
    # Build the file path
    sub_folder = "raw" if raw else "processed"
    path = fr"dataset/ml-{category}/{sub_folder}/ml-{category}.{dimension}"
    
    if not raw and os.path.exists(path):
        return torch.load(path, weights_only=False)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Please ensure the dataset is downloaded and placed correctly.")

    # Load the dataset
    data = pd.read_csv(path, sep='\t', index_col=0)

    # Load pretrained Sentence-T5 model
    model = SentenceTransformer('sentence-transformers/sentence-t5-base')

    # Build textual inputs based on the dimension
    if dimension == "user":
        texts = data.apply(lambda row: f"User is a {row['age:token']}-year-old\
                           {"male" if row['gender:token']=="M" else "female"} \
                           {row['occupation:token']}\
                           living in zip code {row['zip_code:token']}.", axis=1).tolist()
    elif dimension == "item":
        raise NotImplementedError("Item-based text generation not yet implemented.")
    elif dimension == "relation" or dimension == "entity":
        raise NotImplementedError(f"{dimension}-based embeddings not supported yet.")
    else:
        raise ValueError("Invalid dimension. Choose from 'user', 'item', 'relation', or 'entity'.")

    # Generate embeddings
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    torch.save(embeddings, fr"dataset/ml-{category}/processed/ml-{category}.{dimension}")

    torch.cuda.empty_cache()
    return embeddings
    