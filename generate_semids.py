import torch
import argparse
import os
import logging
import pandas as pd
from omegaconf import OmegaConf
from data.loader import load_movie_lens, load_lfm
from modules.rq_vae import RQ_VAE
from modules.rk_means import RKMeans
from modules.rvq import RVQ
from utils.semantic_id_metrics import compute_semantic_id_metrics
from utils.seed import set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data_for_config(config):
    dataset = config.data.dataset
    if dataset == "movielens":
        return load_movie_lens(
            category=config.data.category,
            dimension=config.data.embedding_dimension,
            train=True,
            raw=True,
        )
    elif dataset == "lfm":
        return load_lfm(
            embedding_type=config.data.embedding_dimension,
            normalize_data=config.data.normalize_data,
        )
    else:
        raise ValueError(
            f"Dataset '{dataset}' is not supported. "
            "Extend load_data_for_config() to add support."
        )


def load_trained_model(model_path, config, device, use_rkmeans: bool, use_rvq: bool = False):
    data = load_data_for_config(config)

    if use_rkmeans:
        model = RKMeans(
            input_dim=data.shape[1],
            codebook_size=config.model.codebook_clusters,
            n_quantization_layers=config.model.num_codebook_layers,
        )
    elif use_rvq:
        model = RVQ(
            input_dim=data.shape[1],
            codebook_size=config.model.codebook_clusters,
            n_quantization_layers=config.model.num_codebook_layers,
            commitment_weight=config.model.commitment_weight,
        )
    else:
        model = RQ_VAE(
            input_dim=data.shape[1],
            latent_dim=config.model.latent_dimension,
            hidden_dims=config.model.hidden_dimensions,
            codebook_size=config.model.codebook_clusters,
            n_quantization_layers=config.model.num_codebook_layers,
            commitment_weight=config.model.commitment_weight,
        )

    model.load_state_dict(torch.load(model_path, map_location=device))
    # kmeans_initted is a plain Python attr, not saved in state_dict — set it
    # after loading so the forward pass uses the loaded centroids, not a re-fit.
    for layer in model.quantization_layers:
        layer.kmeans_initted = True
    model.to(device)
    model.eval()
    return model, data


def generate_all_semids(model, data, device, batch_size, use_rkmeans: bool):
    all_semids = []

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size].to(device, dtype=torch.float32)

            if use_rkmeans:
                output = model(batch)
            else:
                output = model.get_semantic_ids(batch)

            all_semids.append(output.sem_ids.cpu())

    return torch.cat(all_semids, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Generate semantic IDs for dataset items")
    parser.add_argument('--config', type=str, default='config/config_lfm_musicnn.yaml')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='outputs/semids.pt')
    parser.add_argument('--temperature', type=float, default=1.35,
                        help='Temperature for Gumbel Softmax — ignored for RKMeans')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--rkmeans', action='store_true',
                        help='Load model as RKMeans instead of RQ-VAE')
    parser.add_argument('--rvq', action='store_true',
                        help='Load model as RVQ instead of RQ-VAE')
    args = parser.parse_args()

    if args.rkmeans and args.rvq:
        parser.error("--rkmeans and --rvq are mutually exclusive")

    config = OmegaConf.load(args.config)
    set_seed(getattr(config.general, "seed", None))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_type = "RKMeans" if args.rkmeans else ("RVQ" if args.rvq else "RQ-VAE")
    logger.info(f"Using device: {device}")
    logger.info(f"Model type: {model_type}")
    logger.info(f"Loading model from: {args.model_path}")
    if not args.rkmeans:
        logger.info(f"Temperature: {args.temperature}")

    model, data = load_trained_model(args.model_path, config, device, args.rkmeans, use_rvq=args.rvq)
    logger.info(f"Loaded {len(data)} items with dimension {data.shape[1]}")

    logger.info("Generating semantic IDs...")
    semids = generate_all_semids(
        model, data, device,
        batch_size=args.batch_size,
        use_rkmeans=args.rkmeans,
    )
    logger.info(f"Generated semantic IDs shape: {semids.shape}")

    metrics = compute_semantic_id_metrics(semids, config.model.codebook_clusters)

    # Save .pt
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save({
        'semantic_ids': semids,
        'metrics': metrics,
        'config': config,
        'num_items': len(data),
        'model_type': 'rkmeans' if args.rkmeans else ('rvq' if args.rvq else 'rq_vae'),
        'temperature': None if args.rkmeans else args.temperature,
    }, args.output_path)
    logger.info(f"Semantic IDs saved to: {args.output_path}")

    # Save .csv
    csv_path = os.path.splitext(args.output_path)[0] + ".csv"
    pd.DataFrame({
        "item_id": range(1, len(semids) + 1),
        "semantic_id": [str(row.tolist()) for row in semids],
    }).to_csv(csv_path, index=False)
    logger.info(f"Semantic IDs CSV saved to: {csv_path}")

    logger.info("Semantic ID statistics:")
    logger.info(f"  Shape:               {semids.shape}")
    logger.info(f"  Min ID per layer:    {semids.min(dim=0)[0].tolist()}")
    logger.info(f"  Max ID per layer:    {semids.max(dim=0)[0].tolist()}")
    logger.info(f"  Global unique IDs:   {len(torch.unique(semids, dim=0))}")
    logger.info(f"  Unique IDs / layer:  {[len(torch.unique(semids[:, i])) for i in range(semids.shape[1])]}")
    logger.info(f"  Global unique ratio: {metrics['unique_ratio']:.4f}")


if __name__ == "__main__":
    main()
