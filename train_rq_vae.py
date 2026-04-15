import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb
import logging
from modules.temperature_scheduler import create_temperature_scheduler
from schemas.quantization import QuantizeForwardMode
from utils.semantic_id_metrics import compute_semantic_id_metrics

logger = logging.getLogger(__name__)


def compute_semid_metrics_on_subset(model, data, device, batch_size, temperature=1.0, max_items=None):
    """Compute semantic ID metrics on a subset for fast logging."""
    model.eval()
    semids_chunks = []

    if max_items is not None:
        data = data[:max_items]

    data_loader = DataLoader(data, batch_size=batch_size, pin_memory=(device.type == "cuda"))
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device, non_blocking=True).float()
            output = model.get_semantic_ids(batch, temperature=temperature)
            semids_chunks.append(output.sem_ids.cpu())

    semids = torch.cat(semids_chunks, dim=0)
    return compute_semantic_id_metrics(semids, codebook_size=model.codebook_size)

def train(model, data, optimizer, scheduler, num_epochs, device, config):
    model.train()

    temperature_annealing = getattr(config.train, 'temperature_annealing', False)
    temperature_update_freq = getattr(config.train, 'temperature_update_frequency', 1)
    quantization_method_str = getattr(config.model, 'quantization_method', 'ste')
    is_gumbel_softmax = quantization_method_str == "gumbel_softmax"

    temperature_scheduler = None
    if temperature_annealing and is_gumbel_softmax:
        annealing_schedule = getattr(config.train, 'annealing_schedule', 'exponential')
        initial_temp = getattr(config.train, 'temperature', 2.0)
        min_temp = getattr(config.train, 'min_temperature', 0.1)
        decay_rate = getattr(config.train, 'temperature_decay', 0.999)

        temperature_scheduler = create_temperature_scheduler(
            schedule_type=annealing_schedule,
            initial_temperature=initial_temp,
            min_temperature=min_temp,
            decay_rate=decay_rate,
            total_steps=num_epochs // temperature_update_freq
        )
        logger.info(f"Temperature annealing: {annealing_schedule} (every {temperature_update_freq} epochs)")

    logger.info(f"Training with {quantization_method_str} quantization")

    epoch_progress = tqdm(range(num_epochs), total=num_epochs, desc="Training Loop")
    results = []

    train_loader = DataLoader(data, batch_size=config.data.batch_size, pin_memory=(device.type == "cuda"))
    validation_step = getattr(config.train, "validation_step", 1)
    if validation_step <= 0:
        validation_step = 1
    global_unique_threshold = getattr(config.train, "global_unique_threshold", 1.0)
    metric_eval_samples = getattr(config.train, "metric_eval_samples", None)
    last_global_unique = None

    for epoch in epoch_progress:
        total_loss = 0
        total_reconstruction_loss = 0
        total_commit_loss = 0
        p_unique = 0

        current_temperature = 1.0
        if temperature_scheduler is not None:
            current_temperature = temperature_scheduler.get_temperature()
            if epoch % temperature_update_freq == 0:
                temperature_scheduler.step()

        if epoch == 0:
            kmeans_init_data = torch.Tensor(data[torch.arange(min(20000, len(data)))]).to(device, dtype=torch.float32)
            model(kmeans_init_data, temperature=current_temperature)

        for batch in train_loader:
            batch = batch.to(device).float()
            optimizer.zero_grad()
            result = model(batch, temperature=current_temperature)
            result.loss.backward()
            optimizer.step()

            total_loss += result.loss.item()
            total_reconstruction_loss += result.reconstruction_loss.item()
            total_commit_loss += result.rqvae_loss.item()
            p_unique += result.p_unique_ids.item()

        epoch_stats = {
            "Epoch": epoch,
            "Loss": total_loss / len(train_loader),
            "Reconstruction Loss": total_reconstruction_loss / len(train_loader),
            "RQ-VAE Loss": total_commit_loss / len(train_loader),
            "Prob Unique IDs": p_unique / len(train_loader)
        }

        computed_global_unique = False
        if epoch % validation_step == 0 or epoch == num_epochs - 1:
            debug = getattr(config.general, "debug", False)

            metrics = compute_semid_metrics_on_subset(
                model=model,
                data=data,
                device=device,
                batch_size=config.data.batch_size,
                temperature=current_temperature,
                max_items=metric_eval_samples,
            )
            global_unique = float(metrics["unique_ratio"])
            epoch_stats["Prob Unique IDs (Global)"] = global_unique
            epoch_stats["Global Unique Ratio"] = global_unique
            last_global_unique = global_unique
            computed_global_unique = True

            per_layer_usage = [float(v) for v in metrics["per_layer_usage"]]
            per_layer_entropy = [float(v) for v in metrics["per_layer_entropy"]]

            for layer_idx, usage_value in enumerate(per_layer_usage):
                epoch_stats[f"Layer Usage/{layer_idx}"] = usage_value
            for layer_idx, entropy_value in enumerate(per_layer_entropy):
                epoch_stats[f"Layer Entropy/{layer_idx}"] = entropy_value

            if debug:
                logger.info(
                    "Semantic ID metrics: unique_ratio=%.4f, usage=%s, entropy=%s",
                    epoch_stats["Global Unique Ratio"],
                    per_layer_usage,
                    per_layer_entropy,
                )
            model.train()

        if is_gumbel_softmax and temperature_scheduler is not None:
            epoch_stats["Temperature"] = current_temperature

        if computed_global_unique and last_global_unique is not None:
            if last_global_unique >= global_unique_threshold:
                logger.info(
                    f"Early stopping at epoch {epoch}: Global unique IDs >= {global_unique_threshold}"
                )
                break

        if config.general.use_wandb:
            wandb.log(epoch_stats, step=epoch)

        epoch_progress.set_postfix(epoch_stats)
        results.append(epoch_stats)

    return results