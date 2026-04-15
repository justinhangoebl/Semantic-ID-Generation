import torch
import wandb
import torch.optim as optim
from torch.optim import lr_scheduler
import math
from train_rq_vae import train
from omegaconf import OmegaConf
from data.loader import load_movie_lens, load_amazon_book, load_lfm
from modules.rq_vae import RQ_VAE
import argparse
import itertools
import json
import os
from datetime import datetime
import random
import numpy as np
import optuna
from optuna.samplers import TPESampler
from schemas.quantization import QuantizeForwardMode, QuantizeDistance


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(config):
    if config.data.dataset == "movielens":
        data = load_movie_lens(
            category=config.data.category,
            dimension=config.data.embedding_dimension,
            train=True,
            raw=True,
        )
    elif config.data.dataset == "amazon_books":
        data = load_amazon_book(
            dimension='meta',
            train=True,
            raw=True
        )
    elif config.data.dataset == "lfm":
        data = load_lfm(
            embedding_type=config.data.embedding_dimension,
            normalize_data=config.data.normalize_data,
        )
    elif config.data.dataset == "lastfm":
        raise NotImplementedError("LastFM dataset loading is not implemented yet.")
    else:
        raise ValueError(f"Unknown dataset: {config.data.dataset}")
    return data


# ---------------------------------------------------------------------------
# Legacy grid / random search helpers (kept for backwards compatibility)
# ---------------------------------------------------------------------------

def generate_random_config(param_grid, num_samples=50):
    """Generate random hyperparameter combinations."""
    configs = []
    for _ in range(num_samples):
        config = {param: random.choice(values) for param, values in param_grid.items()}
        configs.append(config)
    return configs


def generate_grid_search_configs(param_grid, max_combinations=None):
    """Generate all combinations for grid search (use with caution)."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combinations = list(itertools.product(*values))
    if max_combinations and len(all_combinations) > max_combinations:
        all_combinations = random.sample(all_combinations, max_combinations)
    return [dict(zip(keys, combo)) for combo in all_combinations]


def create_hyperparameter_grid():
    """Define hyperparameter search space for legacy random/grid search."""
    return {
        'learning_rate': [1e-4, 5e-4, 1e-3],
        'weight_decay': [0, 1e-4, 1e-3],
        'batch_size': [16, 64, 256, 512],
        'hidden_dimensions': [
            [512, 256, 128],
            [768, 512, 256],
            [1024, 512, 256],
            [768, 384, 192],
        ],
        'latent_dimension': [256],
        'codebook_clusters': [512, 256, 128],
        'num_codebook_layers': [2, 3, 4],
        'commitment_weight': [0.1, 0.25, 0.3],
    }


# ---------------------------------------------------------------------------
# Config / model helpers
# ---------------------------------------------------------------------------

def update_config_with_hyperparams(base_config, hyperparams):
    """Return a new OmegaConf config with hyperparams overlaid."""
    config = OmegaConf.create(base_config)
    config.train.learning_rate = hyperparams['learning_rate']
    config.train.weight_decay = hyperparams['weight_decay']
    config.data.batch_size = hyperparams['batch_size']
    config.model.hidden_dimensions = hyperparams['hidden_dimensions']
    config.model.latent_dimension = hyperparams['latent_dimension']
    config.model.codebook_clusters = hyperparams['codebook_clusters']
    config.model.num_codebook_layers = hyperparams['num_codebook_layers']
    config.model.commitment_weight = hyperparams['commitment_weight']
    config.model.codebook_clusters = hyperparams.get('codebook_clusters', config.model.codebook_clusters)
    config.model.num_codebook_layers = hyperparams.get('num_codebook_layers', config.model.num_codebook_layers)
    config.model.distance_mode = hyperparams.get('distance_mode', 'l2')
    config.model.quantization_method = "gumbel_softmax"  # Fixed for hyperparameter search

    # Temperature annealing (for Gumbel Softmax)
    config.train.temperature_annealing = True
    config.train.temperature_update_frequency = hyperparams.get("temperature_update_frequency", 1)
    config.train.annealing_schedule = hyperparams.get("annealing_schedule", "cosine")

    # Gumbel Softmax parameters
    config.train.temperature = hyperparams.get("temperature", 2.0)
    config.train.min_temperature = hyperparams.get("min_temperature", 0.1)
    config.train.temperature_decay = hyperparams.get("temperature_decay", 0.98)
    config.train.validation_step = hyperparams.get("validation_step", 100)
    return config


def evaluate_model_performance(train_results):
    """Extract key metrics from training results.

    In addition to loss / uniqueness metrics, this function scans backwards
    through the result list to find the most-recent validation step that
    contains per-layer codebook-usage statistics (``Layer Usage/<idx>``).
    Those values are returned as ``codebook_usage_per_layer`` - a list of
    floats in layer order, where 1.0 means every codebook entry was used.
    """
    if not train_results:
        return {
            'final_loss': float('inf'),
            'final_reconstruction_loss': float('inf'),
            'final_rqvae_loss': float('inf'),
            'final_prob_unique_ids': 0.0,
            'final_global_prob_unique_ids': 0.0,
            'avg_loss': float('inf'),
            'convergence_epoch': 0,
            'codebook_usage_per_layer': [],
        }
    final = train_results[-1]
    last_n = max(1, len(train_results) // 10)
    avg_loss = sum(r['Loss'] for r in train_results[-last_n:]) / last_n

    # Find last validation epoch that recorded per-layer usage
    codebook_usage_per_layer = []
    for result in reversed(train_results):
        layer_keys = sorted(
            [k for k in result if k.startswith('Layer Usage/')],
            key=lambda k: int(k.split('/')[1]),
        )
        if layer_keys:
            codebook_usage_per_layer = [result[k] for k in layer_keys]
            break

    return {
        'final_loss': final['Loss'],
        'final_reconstruction_loss': final['Reconstruction Loss'],
        'final_rqvae_loss': final['RQ-VAE Loss'],
        'final_prob_unique_ids': final['Prob Unique IDs'],
        'final_global_prob_unique_ids': final.get('Prob Unique IDs (Global)', 0.0),
        'avg_loss': avg_loss,
        'convergence_epoch': len(train_results),
        'codebook_usage_per_layer': codebook_usage_per_layer,
    }


# ---------------------------------------------------------------------------
# Single-trial training
# ---------------------------------------------------------------------------

def train_single_config(base_config, hyperparams, data, device, trial_id,
                        optuna_trial=None):
    """
    Train one hyperparameter configuration.

    Parameters
    ----------
    optuna_trial : optuna.Trial or None
        When provided, intermediate losses are reported after each epoch so
        Optuna's pruner can terminate unpromising trials early.
    """
    config = update_config_with_hyperparams(base_config, hyperparams)

    if config.general.use_wandb:
        wandb.init(
            project=f"{config.general.wandb_project}_hyperopt",
            entity=config.general.wandb_entity,
            name=f"trial_{trial_id}",
            config=dict(hyperparams),
            reinit=True,
        )

    try:
        # Convert string to enum
        if config.model.quantization_method == "gumbel_softmax":
            quantization_method = QuantizeForwardMode.GUMBEL_SOFTMAX
        elif config.model.quantization_method == "ste":
            quantization_method = QuantizeForwardMode.STE
        else:
            raise ValueError(f"Unknown quantization method: {config.model.quantization_method}")

        # For high-dim inputs (Jukebox ≥ 2048), normalize latent vectors onto
        # the unit sphere before quantization.  This bounds L2 distances to
        # [0, 2] and prevents magnitude variance from collapsing all
        # assignments to one entry.  cosine distance is then redundant (L2 on
        # the sphere is monotone with cosine similarity), so we always use L2.
        normalize_quantizer_inputs = data.shape[1] >= 2048

        model = RQ_VAE(
            input_dim=data.shape[1],
            latent_dim=config.model.latent_dimension,
            hidden_dims=config.model.hidden_dimensions,
            codebook_size=config.model.codebook_clusters,
            codebook_kmeans_init=True,
            codebook_sim_vq=True,
            n_quantization_layers=config.model.num_codebook_layers,
            commitment_weight=config.model.commitment_weight,
            quantization_method=quantization_method,
            distance_mode=QuantizeDistance.L2,
            normalize_quantizer_inputs=normalize_quantizer_inputs,
        )
        model.to(device)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        if config.general.use_wandb:
            wandb.watch(model, log="all")

        # -----------------------------------------------------------------
        # Train – pass the Optuna trial so the loop can report + prune.
        # Your train() function should accept an optional `optuna_trial`
        # kwarg; if it doesn't yet, the kwarg is simply ignored.
        # -----------------------------------------------------------------
        train_results = train(
            model=model,
            data=data,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=config.train.num_epochs,
            device=device,
            config=config,
        )

        performance = evaluate_model_performance(train_results)

        if config.general.use_wandb:
            wandb_metrics = {
                "hp_final_loss": performance['final_loss'],
                "hp_final_reconstruction_loss": performance['final_reconstruction_loss'],
                "hp_final_rqvae_loss": performance['final_rqvae_loss'],
                "hp_final_prob_unique_ids": performance['final_prob_unique_ids'],
                "hp_final_global_prob_unique_ids": performance['final_global_prob_unique_ids'],
                "hp_avg_loss": performance['avg_loss'],
                "hp_convergence_epoch": performance['convergence_epoch'],
            }
            for layer_idx, usage in enumerate(performance.get('codebook_usage_per_layer', [])):
                wandb_metrics[f"hp_codebook_usage_layer_{layer_idx}"] = usage
            wandb.log(wandb_metrics)

        return performance, train_results

    except optuna.exceptions.TrialPruned:
        # Re-raise so Optuna records the trial as pruned, not failed
        raise

    except Exception as e:
        print(f"Error in trial {trial_id}: {e}")
        return {
            'final_loss': float('inf'),
            'final_reconstruction_loss': float('inf'),
            'final_rqvae_loss': float('inf'),
            'final_prob_unique_ids': 0.0,
            'avg_loss': float('inf'),
            'convergence_epoch': 0,
            'error': str(e),
        }, []

    finally:
        if config.general.use_wandb:
            wandb.finish()


# ---------------------------------------------------------------------------
# Optuna objective + study
# ---------------------------------------------------------------------------

def create_optuna_objective(base_config, data, device):
    """
    Factory that closes over base_config / data / device and returns a
    callable objective for optuna.Study.optimize().
    """

    # Pick hidden-dimension search space based on input size
    input_dim = data.shape[1]
    is_jukebox = input_dim >= 2048
    if is_jukebox:
        # Jukebox (4800-dim) – need large intermediate layers
        hidden_dim_choices = [
            '[4096, 2048, 1024]',
            '[2048, 1024, 512]',
            '[4096, 1024, 512]',
            '[2048, 512, 256]',
        ]
    elif input_dim >= 512:
        # Mid-size embeddings (e.g. 768-dim sentence transformers)
        hidden_dim_choices = [
            '[768, 512, 256]',
            '[512, 256, 128]',
            '[1024, 512, 256]',
            '[768, 384, 192]',
        ]
    else:
        # Small embeddings (e.g. MusicNN 50-dim)
        hidden_dim_choices = [
            '[128, 64]',
            '[64, 32]',
            '[32, 16]',
            '[50, 32, 16]',
        ]

    def objective(trial: optuna.Trial) -> float:
        min_temperature = trial.suggest_float("min_temperature", 0.1, 2)
        temperature_low = min_temperature + 0.1
        temperature = trial.suggest_float("temperature", temperature_low, 3.0)

        # Jukebox-specific parameters to combat codebook collapse
        if is_jukebox:
            # Vary codebook granularity and depth independently -
            # collapse is caused by all assignments going to one entry,
            # not by insufficient entries.
            codebook_clusters = trial.suggest_categorical(
                'codebook_clusters', [128, 256]
            )
            num_codebook_layers = 3
            # Higher commitment pushes encoder outputs closer to codebook entries
            commitment_weight = trial.suggest_float('commitment_weight', 0.1, 1)
        else:
            codebook_clusters = 256
            num_codebook_layers = 3
            commitment_weight = trial.suggest_float('commitment_weight', 0.1, 0.75)

        hyperparams = {
            # Continuous on log scale – TPE explores this much more
            # efficiently than a fixed list
            'learning_rate': trial.suggest_float(
                'learning_rate', 1e-6, 1e-3, log=True
            ),
            # Categorical params
            'weight_decay': trial.suggest_categorical(
                'weight_decay', [0, 1e-4, 1e-3]
            ),
            'batch_size': trial.suggest_categorical(
                'batch_size', [256, 512, 1024, 2048]
            ),
            'hidden_dimensions': trial.suggest_categorical(
                'hidden_dimensions',
                # JSON strings so Optuna can hash them; converted below
                hidden_dim_choices,
            ),
            'latent_dimension': trial.suggest_categorical(
                'latent_dimension', [16, 32, 64, 128, 256]
            ),
            'codebook_clusters': codebook_clusters,
            'num_codebook_layers': num_codebook_layers,
            'commitment_weight': commitment_weight,
            'temperature': temperature,
            'min_temperature': min_temperature,
            'temperature_decay': trial.suggest_float(
                'temperature_decay', 0.8, 0.9999999
            ),
            'annealing_schedule': trial.suggest_categorical(
                'annealing_schedule', ["cosine", "exponential", "inverse_log", "constant"]
            ),
        }

        # Optuna stores categoricals as strings; convert back to list
        hyperparams['hidden_dimensions'] = json.loads(
            hyperparams['hidden_dimensions']
        )

        performance, _ = train_single_config(
            base_config, hyperparams, data, device,
            trial_id=trial.number,
            optuna_trial=trial,
        )

        global_unique = performance.get(
            'final_global_prob_unique_ids',
            performance['final_prob_unique_ids']
        )

        # Constraint signal: <= 0 means feasible 
        # Optuna will separate trials into feasible/infeasible and only
        # minimise final_loss among feasible ones. Infeasible trials are
        # still used for exploration but never become the best trial.
        trial.set_user_attr("uniqueness_violation", 0.9 - global_unique)

        # Log everything you already care about
        trial.set_user_attr("global_prob_unique_ids", global_unique)
        trial.set_user_attr("final_loss_raw", performance['final_loss'])
        trial.set_user_attr("rqvae_loss", performance['final_rqvae_loss'])
        trial.set_user_attr("recon_loss", performance['final_reconstruction_loss'])

        # Log per-layer codebook usage so it's visible in Optuna's dashboard
        for layer_idx, usage in enumerate(performance.get('codebook_usage_per_layer', [])):
            trial.set_user_attr(f"codebook_usage_layer_{layer_idx}", usage)

        return performance['final_loss']   # ← pure loss, zero penalty math
    return objective


def hyperparameter_tuning_optuna(config_path, num_trials=50,
                                 output_dir="hyperopt_results",
                                 timeout=None):
    """
    Run Bayesian hyperparameter search with Optuna (TPE + MedianPruner).

    Parameters
    ----------
    timeout : int or None
        Wall-clock limit in seconds. Useful on shared GPU clusters.
    """
    def constraints_func(frozen_trial):
        # Returns a tuple of floats; each value <= 0 means that constraint is satisfied.
        # Optuna only considers a trial "feasible" when ALL values are <= 0.
        return (frozen_trial.user_attrs["uniqueness_violation"],)
    os.makedirs(output_dir, exist_ok=True)

    base_config = OmegaConf.load(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(str(device) + "="*50)

    print("Loading data...")
    data = load_data(base_config)
    print(f"Data shape: {data.shape}")

    sampler = TPESampler(seed=42, multivariate=True,
        constraints_func=constraints_func,)
    # multivariate=True models interactions between params (slightly slower
    # to fit but often more accurate for small trial budgets)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,   # don't prune until we have 5 complete trials
        n_warmup_steps=10,    # don't prune before epoch 10
        interval_steps=1,
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"rqvae_{base_config.data.dataset}",
        # SQLite backend – study survives crashes and can be resumed
        storage=f"sqlite:///{output_dir}/optuna_study.db",
        load_if_exists=True,
    )

    objective = create_optuna_objective(base_config, data, device)

    study.optimize(
        objective,
        n_trials=num_trials,
        timeout=timeout,
        gc_after_trial=True,     # frees GPU memory between trials
        show_progress_bar=True,
    )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    print(f"\n=== Optuna Search Complete ===")
    print(f"Trials finished : {len(study.trials)}")
    print(f"Best trial      : {study.best_trial.number}")
    print(f"Best loss       : {study.best_value:.6f}")
    print("Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    results_summary = {
        'search_type': 'optuna_tpe',
        'num_trials': len(study.trials),
        'best_performance': study.best_value,
        'best_hyperparameters': study.best_params,
        'all_results': [
            {
                'trial_id': t.number,
                'hyperparameters': t.params,
                'performance': {
                    'final_loss': t.value if t.value is not None else float('inf')
                },
                'state': str(t.state),
            }
            for t in study.trials
        ],
        'timestamp': datetime.now().isoformat(),
    }

    results_file = save_results(results_summary, output_dir)

    # ------------------------------------------------------------------
    # Optional visualisations (require optuna[visualization])
    # ------------------------------------------------------------------
    try:
        import optuna.visualization as vis
        fig = vis.plot_optimization_history(study)
        fig.write_html(os.path.join(output_dir, "optimization_history.html"))
        fig = vis.plot_param_importances(study)
        fig.write_html(os.path.join(output_dir, "param_importances.html"))
        fig = vis.plot_parallel_coordinate(study)
        fig.write_html(os.path.join(output_dir, "parallel_coordinate.html"))
        print(f"Visualisations saved to {output_dir}/")
    except ImportError:
        print("Install optuna[visualization] to generate HTML plots.")

    return study


# ---------------------------------------------------------------------------
# Legacy random / grid search entry-point (unchanged)
# ---------------------------------------------------------------------------

def train_single_config_legacy(base_config, hyperparams, data, device, trial_id):
    """Thin wrapper that calls train_single_config without an Optuna trial."""
    return train_single_config(base_config, hyperparams, data, device, trial_id)


def hyperparameter_tuning(config_path, search_type="random", num_trials=50,
                          output_dir="hyperopt_results"):
    """Random / grid search (legacy path)."""
    base_config = OmegaConf.load(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading data...")
    data = load_data(base_config)
    print(f"Data loaded: {data.shape}")

    param_grid = create_hyperparameter_grid()

    if search_type == "random":
        hyperparam_configs = generate_random_config(param_grid, num_trials)
    elif search_type == "grid":
        hyperparam_configs = generate_grid_search_configs(param_grid, num_trials)
    else:
        raise ValueError("search_type must be 'random' or 'grid'")

    print(f"Starting {search_type} search with {len(hyperparam_configs)} configurations...")

    all_results = []
    best_performance = float('inf')
    best_hyperparams = None

    for trial_id, hyperparams in enumerate(hyperparam_configs):
        print(f"\n--- Trial {trial_id + 1}/{len(hyperparam_configs)} ---")
        print(f"Hyperparameters: {hyperparams}")

        performance, train_results = train_single_config_legacy(
            base_config, hyperparams, data, device, trial_id
        )

        result = {
            'trial_id': trial_id,
            'hyperparameters': hyperparams,
            'performance': performance,
            'train_results': train_results[-5:] if train_results else [],
        }
        all_results.append(result)

        if performance['final_loss'] < best_performance:
            best_performance = performance['final_loss']
            best_hyperparams = hyperparams
            print(f"New best performance: {best_performance:.6f}")

        print(f"Current performance: {performance['final_loss']:.6f}")

    results_summary = {
        'search_type': search_type,
        'num_trials': len(hyperparam_configs),
        'best_performance': best_performance,
        'best_hyperparameters': best_hyperparams,
        'all_results': all_results,
        'timestamp': datetime.now().isoformat(),
    }

    results_file = save_results(results_summary, output_dir)

    print(f"\n=== Hyperparameter Tuning Complete ===")
    print(f"Best performance: {best_performance:.6f}")
    print(f"Best hyperparameters: {best_hyperparams}")
    print(f"Results saved to: {results_file}")

    return results_summary


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"hyperparameter_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")
    return results_file


def analyze_results(results_file):
    with open(results_file, 'r') as f:
        results = json.load(f)

    print(f"\n=== Hyperparameter Tuning Analysis ===")
    print(f"Search type: {results['search_type']}")
    print(f"Number of trials: {results['num_trials']}")
    print(f"Best performance: {results['best_performance']:.6f}")
    print("Best hyperparameters:")
    for param, value in results['best_hyperparameters'].items():
        print(f"  {param}: {value}")

    print(f"\n=== Parameter Analysis ===")
    valid_results = [
        r for r in results['all_results']
        if not math.isinf(r['performance']['final_loss'])
    ]

    if len(valid_results) > 5:
        for param in results['best_hyperparameters'].keys():
            values = [r['hyperparameters'][param] for r in valid_results]
            losses = [r['performance']['final_loss'] for r in valid_results]
            if isinstance(values[0], (int, float)):
                correlation = np.corrcoef(values, losses)[0, 1]
                print(f"  {param} correlation with loss: {correlation:.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for RQ-VAE")
    parser.add_argument('--config', type=str, default='config/config_lfm_musicnn.yaml',
                        help='Path to the base configuration file')
    parser.add_argument('--search_type', type=str,
                        choices=['random', 'grid', 'optuna'],
                        default='optuna',
                        help='Type of hyperparameter search')
    parser.add_argument('--num_trials', type=int, default=20,
                        help='Number of trials to run')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Wall-clock timeout in seconds (Optuna only)')
    parser.add_argument('--output_dir', type=str, default='hyperopt_results',
                        help='Directory to save results')
    parser.add_argument('--analyze', type=str, default=None,
                        help='Path to results file to analyze')

    args = parser.parse_args()

    if args.analyze:
        analyze_results(args.analyze)
    elif args.search_type == 'optuna':
        hyperparameter_tuning_optuna(
            config_path=args.config,
            num_trials=args.num_trials,
            output_dir=args.output_dir,
            timeout=args.timeout,
        )
    else:
        hyperparameter_tuning(
            config_path=args.config,
            search_type=args.search_type,
            num_trials=args.num_trials,
            output_dir=args.output_dir,
        )