"""
Training script for RQ-VAE with support for temperature annealing and multiple quantization methods.
"""

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb
import logging
from modules.temperature_scheduler import create_temperature_scheduler
from schemas.quantization import QuantizeForwardMode

logger = logging.getLogger(__name__)

def train(model, data, optimizer, scheduler, num_epochs, device, config):
    """
    Train RQ-VAE model with support for temperature annealing.

    Args:
        model: RQ-VAE model instance
        data: Training data tensor
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler (currently unused but kept for compatibility)
        num_epochs: Number of training epochs
        device: Training device (CPU/GPU)
        config: Configuration object

    Returns:
        list: Training statistics for each epoch
    """
    # Note: scheduler parameter is kept for compatibility but not currently used
    model.train()

    # Get temperature annealing settings
    temperature_annealing = getattr(config.train, 'temperature_annealing', False)
    temperature_update_freq = getattr(config.train, 'temperature_update_frequency', 1)
    quantization_method_str = getattr(config.model, 'quantization_method', 'ste')

    # Convert string to enum for comparison
    is_gumbel_softmax = quantization_method_str == "gumbel_softmax"

    # Initialize temperature scheduler for Gumbel Softmax
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
        logger.info(f"Temperature annealing enabled: {annealing_schedule} schedule (update every {temperature_update_freq} epochs)")

    logger.info(f"Training with {quantization_method_str} quantization")

    epoch_progress = tqdm(range(num_epochs), total=num_epochs, desc="Training Loop")
    results = []

    train_loader = DataLoader(data, batch_size=config.data.batch_size)

    for epoch in epoch_progress:
        total_loss = 0
        total_reconstruction_loss = 0
        total_commit_loss = 0
        p_unique = 0

        # Get current temperature
        current_temperature = 1.0  # Default for STE
        if temperature_scheduler is not None:
            current_temperature = temperature_scheduler.get_temperature()
            # Update temperature according to schedule
            if epoch % temperature_update_freq == 0:
                temperature_scheduler.step()

        # Initialize codebooks on first epoch
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

        # Calculate epoch statistics
        epoch_stats = {
            "Epoch": epoch,
            "Loss": total_loss / len(train_loader),
            "Reconstruction Loss": total_reconstruction_loss / len(train_loader),
            "RQ-VAE Loss": total_commit_loss / len(train_loader),
            "Prob Unique IDs": p_unique / len(train_loader)
        }

        # Add temperature to stats if using Gumbel Softmax
        if is_gumbel_softmax and temperature_scheduler is not None:
            epoch_stats["Temperature"] = current_temperature

        # Early stopping condition
        if p_unique / len(train_loader) >= 1:
            logger.info(f"Early stopping at epoch {epoch}: All IDs are unique")
            break

        if config.general.use_wandb:
            wandb.log(epoch_stats, step=epoch)

        epoch_progress.set_postfix(epoch_stats)
        results.append(epoch_stats)

    return results