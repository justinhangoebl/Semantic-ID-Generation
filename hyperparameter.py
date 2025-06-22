import torch
import wandb
import torch.optim as optim
from torch.optim import lr_scheduler
import math
from train_rq_vae import train
from omegaconf import OmegaConf
from data.loader import load_movie_lens, load_amazon
from modules.rq_vae import RQ_VAE
import argparse
import itertools
import json
import os
from datetime import datetime
import random
import numpy as np


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

def generate_random_config(param_grid, num_samples=50):
    """Generate random hyperparameter combinations"""
    configs = []
    for _ in range(num_samples):
        config = {}
        for param, values in param_grid.items():
            config[param] = random.choice(values)
        configs.append(config)
    return configs

def generate_grid_search_configs(param_grid, max_combinations=None):
    """Generate all combinations for grid search (use with caution - can be very large)"""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    all_combinations = list(itertools.product(*values))
    
    if max_combinations and len(all_combinations) > max_combinations:
        # Randomly sample if too many combinations
        all_combinations = random.sample(all_combinations, max_combinations)
    
    configs = []
    for combination in all_combinations:
        config = dict(zip(keys, combination))
        configs.append(config)
    
    return configs

def create_hyperparameter_grid():
    """Define hyperparameter search space"""
    param_grid = {
        'learning_rate': [5e-4, 1e-3, 2e-3],
        'weight_decay': [1e-5, 1e-4],
        'batch_size': [512],
        'hidden_dimensions': [
            [512, 256, 128],
            [768, 512, 256],
            [1024, 512, 256],
            [768, 384, 192]
        ],
        'latent_dimension': [256],
        'codebook_clusters': [16, 32, 49],
        'num_codebook_layers': [3],
        'commitment_weight': [0.1, 0.25, 0.3]
    }
    return param_grid

def update_config_with_hyperparams(base_config, hyperparams):
    """Update base configuration with hyperparameters"""
    config = OmegaConf.create(base_config)
    
    # Update training parameters
    config.train.learning_rate = hyperparams['learning_rate']
    config.train.weight_decay = hyperparams['weight_decay']
    
    # Update data parameters
    config.data.batch_size = hyperparams['batch_size']
    
    # Update model parameters
    config.model.hidden_dimensions = hyperparams['hidden_dimensions']
    config.model.latent_dimension = hyperparams['latent_dimension']
    config.model.codebook_clusters = hyperparams['codebook_clusters']
    config.model.num_codebook_layers = hyperparams['num_codebook_layers']
    config.model.commitment_weight = hyperparams['commitment_weight']
    
    return config

def evaluate_model_performance(train_results):
    """Extract key metrics from training results for hyperparameter optimization"""
    if not train_results:
        return {
            'final_loss': float('inf'),
            'final_reconstruction_loss': float('inf'),
            'final_rqvae_loss': float('inf'),
            'final_prob_unique_ids': 0.0,
            'avg_loss': float('inf'),
            'convergence_epoch': len(train_results)
        }
    
    final_result = train_results[-1]
    
    # Calculate average loss over last 10% of epochs for stability
    last_10_percent = max(1, len(train_results) // 10)
    recent_results = train_results[-last_10_percent:]
    avg_loss = sum(r['Loss'] for r in recent_results) / len(recent_results)
    
    return {
        'final_loss': final_result['Loss'],
        'final_reconstruction_loss': final_result['Reconstruction Loss'],
        'final_rqvae_loss': final_result['RQ-VAE Loss'],
        'final_prob_unique_ids': final_result['Prob Unique IDs'],
        'avg_loss': avg_loss,
        'convergence_epoch': len(train_results)
    }

def train_single_config(base_config, hyperparams, data, device, trial_id):
    """Train model with specific hyperparameter configuration"""
    config = update_config_with_hyperparams(base_config, hyperparams)
    
    # Initialize wandb for this trial
    if config.general.use_wandb:
        wandb.init(
            project=f"{config.general.wandb_project}_hyperopt",
            entity=config.general.wandb_entity,
            name=f"trial_{trial_id}",
            config=dict(hyperparams),
            reinit=True
        )
    
    try:
        # Create model
        model = RQ_VAE(
            input_dim=data.shape[1],
            latent_dim=config.model.latent_dimension,
            hidden_dims=config.model.hidden_dimensions,
            codebook_size=config.model.codebook_clusters,
            codebook_kmeans_init=True,
            codebook_sim_vq=True,
            n_quantization_layers=config.model.num_codebook_layers,
            commitment_weight=config.model.commitment_weight,
        )
        model.to(device)
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), 
                              lr=config.train.learning_rate, 
                              weight_decay=config.train.weight_decay)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        if config.general.use_wandb:
            wandb.watch(model, log="all")
        
        # Train model
        train_results = train(
            model=model,
            data=data,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=config.train.num_epochs,
            device=device,
            config=config
        )
        
        # Evaluate performance
        performance = evaluate_model_performance(train_results)
        
        # Log final metrics to wandb
        if config.general.use_wandb:
            wandb.log({
                "hp_final_loss": performance['final_loss'],
                "hp_final_reconstruction_loss": performance['final_reconstruction_loss'],
                "hp_final_rqvae_loss": performance['final_rqvae_loss'],
                "hp_final_prob_unique_ids": performance['final_prob_unique_ids'],
                "hp_avg_loss": performance['avg_loss'],
                "hp_convergence_epoch": performance['convergence_epoch']
            })
        
        return performance, train_results
        
    except Exception as e:
        print(f"Error in trial {trial_id}: {str(e)}")
        return {
            'final_loss': float('inf'),
            'final_reconstruction_loss': float('inf'),
            'final_rqvae_loss': float('inf'),
            'final_prob_unique_ids': 0.0,
            'avg_loss': float('inf'),
            'convergence_epoch': 0,
            'error': str(e)
        }, []
    
    finally:
        if config.general.use_wandb:
            wandb.finish()

def save_results(results, output_dir):
    """Save hyperparameter tuning results"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"hyperparameter_results_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_file}")
    return results_file

def hyperparameter_tuning(config_path, search_type="random", num_trials=50, output_dir="hyperopt_results"):
    """Main hyperparameter tuning function"""
    
    # Load base configuration
    base_config = OmegaConf.load(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data once
    print("Loading data...")
    data = load_data(base_config)
    print(f"Data loaded: {data.shape}")
    
    # Generate hyperparameter configurations
    param_grid = create_hyperparameter_grid()
    
    if search_type == "random":
        hyperparam_configs = generate_random_config(param_grid, num_trials)
    elif search_type == "grid":
        hyperparam_configs = generate_grid_search_configs(param_grid, num_trials)
    else:
        raise ValueError("search_type must be 'random' or 'grid'")
    
    print(f"Starting {search_type} search with {len(hyperparam_configs)} configurations...")
    
    # Store results
    all_results = []
    best_performance = float('inf')
    best_hyperparams = None
    
    for trial_id, hyperparams in enumerate(hyperparam_configs):
        print(f"\n--- Trial {trial_id + 1}/{len(hyperparam_configs)} ---")
        print(f"Hyperparameters: {hyperparams}")
        
        # Train model with current hyperparameters
        performance, train_results = train_single_config(
            base_config, hyperparams, data, device, trial_id
        )
        
        # Store results
        result = {
            'trial_id': trial_id,
            'hyperparameters': hyperparams,
            'performance': performance,
            'train_results': train_results[-5:] if train_results else []  # Store last 5 epochs
        }
        all_results.append(result)
        
        # Track best performance
        if performance['final_loss'] < best_performance:
            best_performance = performance['final_loss']
            best_hyperparams = hyperparams
            print(f"New best performance: {best_performance:.6f}")
        
        print(f"Current performance: {performance['final_loss']:.6f}")
    
    # Save all results
    results_summary = {
        'search_type': search_type,
        'num_trials': len(hyperparam_configs),
        'best_performance': best_performance,
        'best_hyperparameters': best_hyperparams,
        'all_results': all_results,
        'timestamp': datetime.now().isoformat()
    }
    
    results_file = save_results(results_summary, output_dir)
    
    print(f"\n=== Hyperparameter Tuning Complete ===")
    print(f"Best performance: {best_performance:.6f}")
    print(f"Best hyperparameters: {best_hyperparams}")
    print(f"Results saved to: {results_file}")
    
    return results_summary

def analyze_results(results_file):
    """Analyze and print summary of hyperparameter tuning results"""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print(f"\n=== Hyperparameter Tuning Analysis ===")
    print(f"Search type: {results['search_type']}")
    print(f"Number of trials: {results['num_trials']}")
    print(f"Best performance: {results['best_performance']:.6f}")
    print(f"Best hyperparameters:")
    for param, value in results['best_hyperparameters'].items():
        print(f"  {param}: {value}")
    
    # Analyze parameter importance (simple correlation analysis)
    print(f"\n=== Parameter Analysis ===")
    all_results = results['all_results']
    
    # Extract valid results (exclude failed trials)
    valid_results = [r for r in all_results if not math.isinf(r['performance']['final_loss'])]
    
    if len(valid_results) > 5:  # Need sufficient data for analysis
        for param in results['best_hyperparameters'].keys():
            values = [r['hyperparameters'][param] for r in valid_results]
            losses = [r['performance']['final_loss'] for r in valid_results]
            
            # Simple correlation analysis for numeric parameters
            if isinstance(values[0], (int, float)):
                correlation = np.corrcoef(values, losses)[0, 1]
                print(f"  {param} correlation with loss: {correlation:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for RQ-VAE")
    parser.add_argument('--config', type=str, default='config/config_ml1m.yaml', 
                       help='Path to the base configuration file')
    parser.add_argument('--search_type', type=str, choices=['random', 'grid'], default='random',
                       help='Type of hyperparameter search')
    parser.add_argument('--num_trials', type=int, default=20,
                       help='Number of trials to run')
    parser.add_argument('--output_dir', type=str, default='hyperopt_results',
                       help='Directory to save results')
    parser.add_argument('--analyze', type=str, default=None,
                       help='Path to results file to analyze')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_results(args.analyze)
    else:
        hyperparameter_tuning(
            config_path=args.config,
            search_type=args.search_type,
            num_trials=args.num_trials,
            output_dir=args.output_dir
        )