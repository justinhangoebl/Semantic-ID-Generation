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

def load_movie_lens(category='1M', dimension="user", train=True, raw=True):
    # Build the file path
    sub_folder = "raw" if raw else "processed"
    path = fr"dataset/ml-{category}/{sub_folder}/ml-{category}.{dimension}"

    if not raw and os.path.exists(path):
        return torch.load(path, weights_only=False)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Please ensure the dataset is downloaded and placed correctly.")

    # Load the dataset
    data = pd.read_csv(path, sep='\t', index_col=0)
    print(data.columns)

    # Load pretrained Sentence-T5 model
    model = SentenceTransformer('sentence-transformers/sentence-t5-base')

    # Build textual inputs based on the dimension
    if dimension == "user":
        texts = data.apply(lambda row: f"User is a {row['age:token']}-year-old\
                           {"male" if row['gender:token']=="M" else "female"} \
                           {row['occupation:token']}\
                           living in zip code {row['zip_code:token']}.", axis=1).tolist()
    elif dimension == "item":
        # texts = data.apply(lambda row: f"The movie '{row['movie_title:token_seq']}' ({row['release_year:token']})\
        #     belongs to the following genres: {row['genre:token_seq']}.", axis=1).tolist()
        texts = data.apply(lambda row: f"{row['movie_title:token_seq']} ({row['release_year:token']}) with genres: {row['genre:token_seq'].lower()}", axis=1).tolist()

    elif dimension == "relation":
        texts = data.apply(lambda row: f"{row['relation_nl:token']}", axis=1).tolist()
    elif dimension == "entity":
        raise NotImplementedError(f"{dimension}-based embeddings not supported yet.")
    else:
        raise ValueError("Invalid dimension. Choose from 'user', 'item', 'relation', or 'entity'.")

    # Generate embeddings
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    torch.save(embeddings, fr"dataset/ml-{category}/processed/ml-{category}.{dimension}")

    torch.cuda.empty_cache()
    return embeddings

def load_last_fm(category='1B', dimension="item", train=True, raw=True):
    # Build the file path
    sub_folder = "raw" if raw else "processed"
    path = fr"dataset/ml-{category}/{sub_folder}/ml-{category}.{dimension}"

    if not raw and os.path.exists(path):
        return torch.load(path, weights_only=False)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Please ensure the dataset is downloaded and placed correctly.")

    # Load the dataset
    data = pd.read_csv(path, sep='\t', index_col=0)
    print(data.columns)

    # Load pretrained Sentence-T5 model
    model = SentenceTransformer('sentence-transformers/sentence-t5-base')

    # Build textual inputs based on the dimension
    if dimension == "user":
        texts = data.apply(lambda row: f"User is a {row['age:token']}-year-old\
                           {"male" if row['gender:token']=="M" else "female"} \
                           {row['occupation:token']}\
                           living in zip code {row['zip_code:token']}.", axis=1).tolist()
    elif dimension == "item":
        # texts = data.apply(lambda row: f"The movie '{row['movie_title:token_seq']}' ({row['release_year:token']})\
        #     belongs to the following genres: {row['genre:token_seq']}.", axis=1).tolist()
        texts = data.apply(lambda row: f"{row['movie_title:token_seq']} ({row['release_year:token']}): {row['genre:token_seq']}", axis=1).tolist()

    elif dimension == "relation":
        texts = data.apply(lambda row: f"{row['relation_nl:token']}", axis=1).tolist()
    elif dimension == "entity":
        raise NotImplementedError(f"{dimension}-based embeddings not supported yet.")
    else:
        raise ValueError("Invalid dimension. Choose from 'user', 'item', 'relation', or 'entity'.")

    # Generate embeddings
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    torch.save(embeddings, fr"dataset/ml-{category}/processed/ml-{category}.{dimension}")

    torch.cuda.empty_cache()
    return embeddings


def load_lfm(embedding_type="jukebox", normalize_data=False):
    """Load lfm music dataset pre-computed embeddings (Jukebox or MusicNN).

    The .jukebox file contains 4800-dim embeddings; .musicnn has 50-dim embeddings.
    Both files use a string track-ID as the first column (``id``), followed by
    float embedding dimensions.  Rows are ordered consistently with lfm.item,
    so row index i corresponds to item_id i.

    Args:
        embedding_type: ``"jukebox"`` or ``"musicnn"``.
        normalize_data: If True, L2-normalise each embedding vector.

    Returns:
        torch.Tensor of shape (num_items, embedding_dim).
    """
    path = fr"dataset/lfm/lfm.{embedding_type}"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"lfm {embedding_type} embeddings not found at {path}. "
            "Please ensure the dataset is placed in dataset/lfm/."
        )

    data = pd.read_csv(path, sep='\t', index_col=0)
    embeddings = torch.tensor(data.values, dtype=torch.float32)

    if normalize_data:
        embeddings = torch.tensor(normalize(embeddings.numpy()), dtype=torch.float32)

    return embeddings


def load_amazon_book(dimension="meta", train=True, raw=True):
    # Build the file path
    raw_folder = "raw" if train else "processed"
    path = fr"dataset/amazon_books/{raw_folder}/amazon_books.{dimension}"

    if not raw and os.path.exists(path):
        return torch.load(path, weights_only=False)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Please ensure the dataset is downloaded and placed correctly.")

    # Load the dataset
    data = pd.read_csv(path, sep='\t')
    print(data.columns)

    # Load pretrained Sentence-T5 model
    model = SentenceTransformer('sentence-transformers/sentence-t5-base')

    texts = data.apply(lambda row: f"{row['title']} talks about {row['description']}, was written in these categories: {row['category']}, and ranks {row['rank']}", axis=1).tolist()
    # Generate embeddings
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    path = fr"dataset/amazon_books/processed/amazon_books.{dimension}"

    torch.cuda.empty_cache()
    return embeddings
