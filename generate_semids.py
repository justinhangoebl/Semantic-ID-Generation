import torch
import argparse
from omegaconf import OmegaConf
from data.loader import load_movie_lens
from modules.rq_vae import RQ_VAE
import os
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_trained_model(model_path, config, device):
    """Load a trained RQ-VAE model."""
    # Get input dimension from data
    data = load_movie_lens(
        category=config.data.category,
        dimension=config.data.embedding_dimension,
        train=True,
        raw=True
    )
    
    model = RQ_VAE(
        input_dim=data.shape[1],
        latent_dim=config.model.latent_dimension,
        hidden_dims=config.model.hidden_dimensions,
        codebook_size=config.model.codebook_clusters,
        n_quantization_layers=config.model.num_codebook_layers,
        commitment_weight=config.model.commitment_weight,
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Mark all quantization layers as initialized to prevent k-means reinit
    for layer in model.quantization_layers:
        layer.kmeans_initted = True
    
    return model

def generate_all_semids(model, data, device, batch_size=64, temperature=1.0):
    """Generate semantic IDs for all items in the dataset."""
    model.eval()
    all_semids = []
    
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size].to(device, dtype=torch.float32)
            
            # Get semantic IDs for the batch
            output = model.get_semantic_ids(batch, temperature=temperature)
            semids = output.sem_ids  # Shape: (batch_size, num_layers)
            
            all_semids.append(semids.cpu())
    
    return torch.cat(all_semids, dim=0)

def main():
    parser = argparse.ArgumentParser(description="Generate semantic IDs for ML-100k items")
    parser.add_argument('--config', type=str, default='config/config_ml100k_item.yaml',
                       help='Path to the configuration file')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model file')
    parser.add_argument('--output_path', type=str, default='outputs/ml100k_semids.pt',
                       help='Path to save the semantic IDs')
    parser.add_argument('--temperature', type=float, default=0.5,
                       help='Temperature for Gumbel Softmax (lower = sharper)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for processing')
    
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Using device: {device}")
    logger.info(f"Loading model from: {args.model_path}")
    
    # Load trained model
    model = load_trained_model(args.model_path, config, device)
    
    # Load all items data
    data = load_movie_lens(
        category=config.data.category,
        dimension=config.data.embedding_dimension,
        train=True,
        raw=True
    )
    
    logger.info(f"Loaded {len(data)} items with dimension {data.shape[1]}")
    
    # Generate semantic IDs
    logger.info("Generating semantic IDs...")
    semids = generate_all_semids(
        model, data, device, 
        batch_size=args.batch_size,
        temperature=args.temperature
    )
    
    logger.info(f"Generated semantic IDs shape: {semids.shape}")
    
    # Save results
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save({
        'semantic_ids': semids,
        'config': config,
        'num_items': len(data),
        'temperature': args.temperature
    }, args.output_path)
    
    logger.info(f"Semantic IDs saved to: {args.output_path}")
    
    # Print some statistics
    logger.info(f"Semantic ID statistics:")
    logger.info(f"  Shape: {semids.shape}")
    logger.info(f"  Min ID per layer: {semids.min(dim=0)[0]}")
    logger.info(f"  Max ID per layer: {semids.max(dim=0)[0]}")
    logger.info(f"  Unique IDs per layer: {[len(torch.unique(semids[:, i])) for i in range(semids.shape[1])]}")

if __name__ == "__main__":
    main()

