import torch


def compute_semantic_id_metrics(sem_ids: torch.Tensor, codebook_size: int) -> dict:
    """Compute global uniqueness and per-layer usage/entropy metrics.

    Args:
        sem_ids: Tensor of shape (num_items, num_layers).
        codebook_size: Number of codebook entries per layer.

    Returns:
        dict with unique_ratio, per_layer_usage, per_layer_entropy.
    """
    if sem_ids.numel() == 0:
        return {
            "unique_ratio": torch.tensor(0.0),
            "per_layer_usage": [],
            "per_layer_entropy": [],
        }

    unique_ratio = torch.unique(sem_ids, dim=0).shape[0] / sem_ids.shape[0]
    per_layer_usage = []
    per_layer_entropy = []

    for layer in range(sem_ids.shape[1]):
        ids = sem_ids[:, layer]
        counts = torch.bincount(ids, minlength=codebook_size).float()
        probs = counts / counts.sum().clamp_min(1.0)
        usage = (counts > 0).float().mean()
        entropy = -(probs * (probs + 1e-9).log()).sum() / torch.log(torch.tensor(codebook_size, dtype=torch.float))
        per_layer_usage.append(usage)
        per_layer_entropy.append(entropy)

    return {
        "unique_ratio": unique_ratio,
        "per_layer_usage": per_layer_usage,
        "per_layer_entropy": per_layer_entropy,
    }
