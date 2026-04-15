import argparse
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def perform_pca_on_latent_space(latent_vectors, n_components=2, random_state=42):
    """
    Perform PCA on the latent space vectors.

    Args:
        latent_vectors (np.ndarray): The latent vectors to be reduced.
        n_components (int): The number of principal components to compute.
        random_state (int): Seed for reproducible PCA initialization.

    Returns:
        np.ndarray: The transformed latent vectors in the PCA space.
    """
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=random_state)
    pca_result = pca.fit_transform(latent_vectors)
    return pca_result


def load_onion_jukebox(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Onion Jukebox embeddings not found at {path}. "
            "Please ensure the dataset is placed in dataset/onion/."
        )
    return pd.read_csv(path, sep="\t", index_col=0)


def save_pca_embeddings(index, pca_vectors, output_path):
    columns = [f"pca_{i}" for i in range(pca_vectors.shape[1])]
    df = pd.DataFrame(pca_vectors, index=index, columns=columns)
    df.to_csv(output_path, sep="\t")


def parse_args():
    parser = argparse.ArgumentParser(description="PCA for Onion Jukebox embeddings")
    parser.add_argument(
        "--input",
        type=str,
        default="dataset/onion/onion.jukebox",
        help="Path to Onion Jukebox embeddings (.jukebox)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset/onion/onion.jukebox_pca768",
        help="Path to save PCA-reduced embeddings",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=768,
        help="Number of PCA components",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for PCA",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    embeddings_df = load_onion_jukebox(args.input)
    latent_vectors = embeddings_df.values.astype(np.float32)
    pca_result = perform_pca_on_latent_space(
        latent_vectors,
        n_components=args.n_components,
        random_state=args.random_state,
    )
    save_pca_embeddings(embeddings_df.index, pca_result, args.output)
    print(
        f"Saved PCA embeddings to {args.output} with shape {pca_result.shape}."
    )


if __name__ == "__main__":
    main()
    