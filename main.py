"""
Main training script for RQ-VAE with support for both STE and Gumbel Softmax quantization.

This script provides a command-line interface for training RQ-VAE models on various datasets
with configurable quantization methods and hyperparameters.
"""

import torch
import wandb
import torch.optim as optim
from torch.optim import lr_scheduler
from utils.wandb import wandb_init
from train_rq_vae import train
from omegaconf import OmegaConf
from data.loader import load_movie_lens, load_amazon
from modules.rq_vae import RQ_VAE
from utils.model_id_generation import generate_model_id
from schemas.quantization import QuantizeForwardMode
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(config):
    """
    Load dataset based on configuration.

    Args:
        config: OmegaConf configuration object

    Returns:
        torch.Tensor: Loaded dataset

    Raises:
        NotImplementedError: For unsupported datasets
        ValueError: For unknown dataset names
    """
    if config.data.dataset == "movielens":
        data = load_movie_lens(
            category=config.data.category,
            dimension=config.data.embedding_dimension,
            train=True,
            raw=True
        )
    elif config.data.dataset == "amazon":
        data = load_amazon(
            category=config.data.category,
            normalize_data=config.data.normalize_data,
            train=True
        )
    elif config.data.dataset == "lastfm":
        raise NotImplementedError("LastFM dataset loading is not implemented yet.")
    else:
        raise ValueError(f"Unknown dataset: {config.data.dataset}")

    logger.info(f"Loaded {config.data.dataset} dataset with shape: {data.shape}")
    return data

def create_model(config, input_dim):
    """
    Create RQ-VAE model with configuration parameters.

    Args:
        config: OmegaConf configuration object
        input_dim: Input dimension of the data

    Returns:
        RQ_VAE: Configured model instance
    """
    # Get quantization parameters with defaults
    quantization_method_str = getattr(config.model, 'quantization_method', 'ste')

    # Convert string to enum
    if quantization_method_str == "gumbel_softmax":
        quantization_method = QuantizeForwardMode.GUMBEL_SOFTMAX
    elif quantization_method_str == "ste":
        quantization_method = QuantizeForwardMode.STE
    else:
        raise ValueError(f"Unknown quantization method: {quantization_method_str}")

    model = RQ_VAE(
        input_dim=input_dim,
        latent_dim=config.model.latent_dimension,
        hidden_dims=config.model.hidden_dimensions,
        codebook_size=config.model.codebook_clusters,
        codebook_kmeans_init=True,
        codebook_sim_vq=True,
        n_quantization_layers=config.model.num_codebook_layers,
        commitment_weight=config.model.commitment_weight,
        quantization_method=quantization_method,
    )

    logger.info(f"Created RQ-VAE model with {quantization_method_str} quantization")
    return model

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train RQ-VAE with configurable quantization methods")
    parser.add_argument('--config', type=str, default='config/config_ml1m_item.yaml',
                       help='Path to the configuration file')
    args = parser.parse_args()

    # Load configuration
    config = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = generate_model_id(config)

    logger.info(f"Using device: {device}")
    logger.info(f"Model ID: {model_id}")

    # Initialize wandb if enabled
    if config.general.use_wandb:
        wandb_init(config)

    # Load data and create model
    data = load_data(config)
    model = create_model(config, data.shape[1])
    model.to(device)

    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(),
                           lr=config.train.learning_rate,
                           weight_decay=config.train.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Watch model with wandb if enabled
    if config.general.use_wandb:
        wandb.watch(model, log="all")

    # Train model
    logger.info("Starting training...")
    train_results = train(
        model=model,
        data=data,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config.train.num_epochs,
        device=device,
        config=config
    )

    # Save model
    torch.save(model.state_dict(), f"models/{model_id}.pt")
    logger.info(f"Training completed. Final results: {train_results[-1]}")

    if config.general.use_wandb:
        wandb.finish()
    
if __name__ == "__main__":
    main()