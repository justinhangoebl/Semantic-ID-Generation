import torch
import wandb
import logging
from torch.utils.data import DataLoader
from utils.semantic_id_metrics import compute_semantic_id_metrics

logger = logging.getLogger(__name__)


def compute_semid_metrics_on_subset(model, data, device, batch_size, max_items=None):
    """Compute semantic ID metrics on a subset for logging."""
    model.eval()
    semids_chunks = []

    if max_items is not None:
        data = data[:max_items]

    pin_memory = device.type == "cuda" and data.device.type == "cpu"
    data_loader = DataLoader(data, batch_size=batch_size, pin_memory=pin_memory)

    for batch in data_loader:                              # forward is already @no_grad
        batch = batch.to(device, non_blocking=True).float()
        output = model(batch)
        semids_chunks.append(output.sem_ids.cpu())

    semids = torch.cat(semids_chunks, dim=0)
    return compute_semantic_id_metrics(semids, codebook_size=model.codebook_size)


def train(model, data, config):
    """
    RKMeans 'training' — fits codebooks once via K-Means, then evaluates.

    No optimizer, no epochs, no loss, no temperature annealing.
    Everything that was specific to gradient-based RQ-VAE training is removed:
      - epoch loop              → gone (one-shot K-Means fit)
      - optimizer / scheduler   → gone (no parameters to update)
      - reconstruction loss     → gone (no decoder)
      - commitment loss         → gone (no encoder gradients)
      - temperature annealing   → gone (no Gumbel-Softmax)
      - early stopping on loss  → gone (replaced by metric logging)
    """
    device = next(model.parameters()).device

    if device.type == "cuda":
        data = data.to(device).float()
    kmeans_init_samples = getattr(config.train, "kmeans_init_samples", 20_000)
    init_data = data[:min(kmeans_init_samples, len(data))]

    logger.info(f"Fitting {model.n_quantization_layers} codebooks "
                f"(size {model.codebook_size}) on {len(init_data)} samples …")

    model.kmeans_init_codebooks(init_data)
    logger.info("Codebook fitting complete — codebooks are now frozen.")

    metric_eval_samples = getattr(config.train, "metric_eval_samples", None)
    metrics = compute_semid_metrics_on_subset(
        model=model,
        data=data,
        device=device,
        batch_size=config.data.batch_size,
        max_items=metric_eval_samples,
    )

    per_layer_usage   = [float(v) for v in metrics["per_layer_usage"]]
    per_layer_entropy = [float(v) for v in metrics["per_layer_entropy"]]
    global_unique     = float(metrics["unique_ratio"])

    results = {
        "Global Unique Ratio": global_unique,
        **{f"Layer Usage/{i}":   v for i, v in enumerate(per_layer_usage)},
        **{f"Layer Entropy/{i}": v for i, v in enumerate(per_layer_entropy)},
    }

    debug = getattr(config.general, "debug", False)
    if debug:
        logger.info(
            "Semantic ID metrics: unique_ratio=%.4f, usage=%s, entropy=%s",
            global_unique, per_layer_usage, per_layer_entropy,
        )

    if config.general.use_wandb:
        wandb.log(results, step=0)

    return [results]