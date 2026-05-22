import os
import torch
import pandas as pd
from transformers import T5EncoderModel, AutoTokenizer
from sklearn.preprocessing import normalize
from tqdm import tqdm


def _encode_texts_t5(texts, model_name='sentence-transformers/sentence-t5-base', batch_size=64, device=None):
    """Encode texts with T5 encoder + mean pooling (GRID-style). No output normalisation."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5EncoderModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc='Encoding with T5'):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
        token_embs = outputs.last_hidden_state          # (B, L, D)
        mask = encoded['attention_mask'].unsqueeze(-1).float()
        embeddings = (token_embs * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        all_embeddings.append(embeddings.cpu())

    torch.cuda.empty_cache()
    return torch.cat(all_embeddings, dim=0)


def load_movie_lens(category='1M', dimension='item', train=True, raw=True):
    sub_folder = 'raw' if raw else 'processed'
    path = fr'dataset/ml-{category}/{sub_folder}/ml-{category}.{dimension}'
    cache_path = fr'dataset/ml-{category}/processed/ml-{category}.{dimension}.pt'

    if os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=False)

    if not os.path.exists(path):
        raise FileNotFoundError(f'Dataset not found at {path}.')

    data = pd.read_csv(path, sep='\t', index_col=0)

    if dimension == 'user':
        texts = data.apply(
            lambda row: (
                f"User is a {row['age:token']}-year-old "
                f"{'male' if row['gender:token'] == 'M' else 'female'} "
                f"{row['occupation:token']} living in zip code {row['zip_code:token']}."
            ), axis=1
        ).tolist()
    elif dimension == 'item':
        texts = data.apply(
            lambda row: f"{row['movie_title:token_seq']} ({row['release_year:token']}) with genres: {row['genre:token_seq'].lower()}",
            axis=1
        ).tolist()
    elif dimension == 'relation':
        texts = data.apply(lambda row: f"{row['relation_nl:token']}", axis=1).tolist()
    else:
        raise ValueError(f"Invalid dimension '{dimension}'. Choose from 'user', 'item', 'relation'.")

    embeddings = _encode_texts_t5(texts)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(embeddings, cache_path)
    return embeddings


def load_lfm(embedding_type='jukebox', normalize_data=False):
    """Load pre-computed LFM embeddings (Jukebox or MusicNN)."""
    path = fr'dataset/lfm/lfm.{embedding_type}'
    if not os.path.exists(path):
        raise FileNotFoundError(f'LFM {embedding_type} embeddings not found at {path}.')

    data = pd.read_csv(path, sep='\t', index_col=0)
    embeddings = torch.tensor(data.values, dtype=torch.float32)

    if normalize_data:
        embeddings = torch.tensor(normalize(embeddings.numpy()), dtype=torch.float32)

    return embeddings


def load_amazon_book(dimension='meta', train=True, raw=True):
    raw_folder = 'raw' if train else 'processed'
    path = fr'dataset/amazon_books/{raw_folder}/amazon_books.{dimension}'
    cache_path = fr'dataset/amazon_books/processed/amazon_books.{dimension}.pt'

    if os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=False)

    if not os.path.exists(path):
        raise FileNotFoundError(f'Dataset not found at {path}.')

    data = pd.read_csv(path, sep='\t')
    texts = data.apply(
        lambda row: (
            f"{row['title']} talks about {row['description']}, "
            f"was written in these categories: {row['category']}, "
            f"and ranks {row['rank']}"
        ), axis=1
    ).tolist()

    embeddings = _encode_texts_t5(texts)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(embeddings, cache_path)
    return embeddings
