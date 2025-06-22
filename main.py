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
import argparse

def load_data(config):
    if config.data.dataset == "movielens":
        data = load_movie_lens(category=config.data.category, 
                                dimension=config.data.embedding_dimension, 
                                train=True,
                                raw=True)
    elif config.data.dataset == "amazon":
        data = load_amazon(
                            category=config.data.category,
                            normalize_data=config.data.normalize_data,
                            train=True)
    elif config.data.dataset == "lastfm":
        raise NotImplementedError("LastFM dataset loading is not implemented yet.")
    else:
        raise ValueError(f"Unknown dataset: {config.data.dataset}")
    
    return data

def main():
    
    parser = argparse.ArgumentParser(description="Train RQ-VAE on MovieLens dataset")
    parser.add_argument('--config', type=str, default='config/config_ml1m.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = generate_model_id(config)
    
    if config.general.use_wandb:
        wandb_init(config)
    
    data = load_data(config)
    
    model = RQ_VAE(
        input_dim = data.shape[1],
        latent_dim = config.model.latent_dimension,
        hidden_dims = config.model.hidden_dimensions,
        codebook_size = config.model.codebook_clusters,
        codebook_kmeans_init = True,
        codebook_sim_vq = True,
        n_quantization_layers = config.model.num_codebook_layers,
        commitment_weight = config.model.commitment_weight,
    )
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    if(config.general.use_wandb):
        wandb.watch(model, log="all")
    
    print(model_id)
    
    train_results = train(
        model=model,
        data=data,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config.train.num_epochs,
        device=device,
        config=config
    )
    torch.save(model.state_dict(), f"models/{model_id}.pt")
    print("Training completed."
          " Results:", train_results[-1])
    
if __name__ == "__main__":
    main()