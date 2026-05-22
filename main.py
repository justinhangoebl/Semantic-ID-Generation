import sys
import torch
import wandb
import torch.optim as optim
from torch.optim import lr_scheduler
from utils.wandb import wandb_init
from train_rq_vae import train as train_rqvae
from train_rk_means import train as train_rkmeans
from train_rvq import train as train_rvq
from omegaconf import OmegaConf
from data.loader import load_movie_lens, load_lfm
from modules.rq_vae import RQ_VAE
from modules.rk_means import RKMeans
from modules.rvq import RVQ
from utils.model_id_generation import generate_model_id
import argparse
import logging
from utils.seed import set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(config):
    if config.data.dataset == 'movielens':
        data = load_movie_lens(
            category=config.data.category,
            dimension=config.data.embedding_dimension,
            train=True,
            raw=True,
        )
    elif config.data.dataset == 'lfm':
        data = load_lfm(
            embedding_type=config.data.embedding_dimension,
            normalize_data=config.data.normalize_data,
        )
    else:
        raise ValueError(f'Unknown dataset: {config.data.dataset}')

    logger.info(f'Loaded {config.data.dataset} dataset with shape: {data.shape}')
    return data


def create_model(config, input_dim, use_rkmeans: bool, use_rvq: bool = False):
    if use_rkmeans:
        model = RKMeans(
            input_dim=input_dim,
            codebook_size=config.model.codebook_clusters,
            n_quantization_layers=config.model.num_codebook_layers,
        )
        logger.info('Created RKMeans model')
        return model

    if use_rvq:
        model = RVQ(
            input_dim=input_dim,
            codebook_size=config.model.codebook_clusters,
            n_quantization_layers=config.model.num_codebook_layers,
            commitment_weight=config.model.commitment_weight,
        )
        logger.info('Created RVQ model')
        return model

    model = RQ_VAE(
        input_dim=input_dim,
        latent_dim=config.model.latent_dimension,
        hidden_dims=config.model.hidden_dimensions,
        codebook_size=config.model.codebook_clusters,
        n_quantization_layers=config.model.num_codebook_layers,
        commitment_weight=config.model.commitment_weight,
    )
    logger.info('Created RQ-VAE model')
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config_lfm_mfcc.yaml')
    parser.add_argument('--rkmeans', action='store_true')
    parser.add_argument('--rvq', action='store_true')
    args = parser.parse_args()

    if args.rkmeans and args.rvq:
        parser.error('--rkmeans and --rvq are mutually exclusive')

    config = OmegaConf.load(args.config)
    set_seed(getattr(config.general, 'seed', None))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_id = generate_model_id(config)

    model_type = 'RKMeans' if args.rkmeans else ('RVQ' if args.rvq else 'RQ-VAE')
    logger.info(f'Device: {device} | Model: {model_type} | ID: {model_id}')

    data = load_data(config)

    if config.general.use_wandb:
        wandb_init(config)

    model = create_model(config, data.shape[1], use_rkmeans=args.rkmeans, use_rvq=args.rvq)
    model.to(device)
    if device.type == 'cuda' and hasattr(torch, 'compile') and sys.platform != 'win32':
        model = torch.compile(model)

    if args.rkmeans:
        train_results = train_rkmeans(model=model, data=data, config=config)
    elif args.rvq:
        optimizer = optim.Adagrad(
            model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.num_epochs, eta_min=1e-6)
        if config.general.use_wandb:
            wandb.watch(model, log=None)
        train_results = train_rvq(
            model=model, data=data, optimizer=optimizer, scheduler=scheduler,
            num_epochs=config.train.num_epochs, device=device, config=config,
        )
    else:
        optimizer = optim.Adagrad(
            model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
        warmup_epochs = getattr(config.train, 'warmup_epochs', 50)
        total_epochs = config.train.num_epochs
        min_lr_ratio = getattr(config.train, 'min_lr_ratio', 0.01)

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / max(1, warmup_epochs)
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return max(min_lr_ratio, 1.0 - (1.0 - min_lr_ratio) * progress)

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        if config.general.use_wandb:
            wandb.watch(model, log=None)
        train_results = train_rqvae(
            model=model, data=data, optimizer=optimizer, scheduler=scheduler,
            num_epochs=config.train.num_epochs, device=device, config=config,
        )

    torch.save(model.state_dict(), f'models/{model_id}.pt')
    logger.info(f'Training completed. Final: {train_results[-1]}')

    if config.general.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
