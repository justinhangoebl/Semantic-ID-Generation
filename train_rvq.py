import torch
from tqdm import tqdm
import wandb
import logging
from utils.semantic_id_metrics import compute_semantic_id_metrics

logger = logging.getLogger(__name__)


def compute_semid_metrics_on_subset(model, data, device, batch_size, max_items=None):
    model.eval()
    if max_items is not None:
        data = data[:max_items]
    data = data.to(device).float()
    semids_chunks = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            output = model.get_semantic_ids(data[i:i + batch_size])
            semids_chunks.append(output.sem_ids.cpu())
    semids = torch.cat(semids_chunks, dim=0)
    return compute_semantic_id_metrics(semids, codebook_size=model.codebook_size)


def train(model, data, optimizer, scheduler, num_epochs, device, config):
    model.train()

    if device.type == 'cuda':
        data = data.to(device).float()

    seed = getattr(config.general, 'seed', None)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=data.device)
        generator.manual_seed(seed)

    batch_size = config.data.batch_size
    n_batches = (len(data) + batch_size - 1) // batch_size

    def iter_batches():
        perm = torch.randperm(len(data), device=data.device, generator=generator)
        for i in range(0, len(data), batch_size):
            yield data[perm[i:i + batch_size]]

    validation_step = getattr(config.train, 'validation_step', 1)
    if validation_step <= 0:
        validation_step = 1
    global_unique_threshold = getattr(config.train, 'global_unique_threshold', 1.0)
    metric_eval_samples = getattr(config.train, 'metric_eval_samples', None)

    epoch_progress = tqdm(range(num_epochs), total=num_epochs, desc='Training RVQ')
    results = []

    for epoch in epoch_progress:
        if epoch == 0:
            model(data[:min(20000, len(data))].to(device).float())

        total_loss = torch.zeros(1, device=device)
        p_unique = torch.zeros(1, device=device)

        model.train()
        for batch in iter_batches():
            optimizer.zero_grad(set_to_none=True)
            result = model(batch)
            result.loss.backward()
            optimizer.step()

            total_loss += result.loss.detach()
            p_unique += result.p_unique_ids.detach()

        if scheduler is not None:
            scheduler.step()

        epoch_stats = {
            'L': (total_loss / n_batches).item(),
            'P_u': (p_unique / n_batches).item(),
        }

        computed_global_unique = False
        if epoch % validation_step == 0 or epoch == num_epochs - 1:
            metrics = compute_semid_metrics_on_subset(
                model=model, data=data, device=device,
                batch_size=batch_size, max_items=metric_eval_samples,
            )
            global_unique = float(metrics['unique_ratio'])
            epoch_stats['Global Unique Ratio'] = global_unique
            computed_global_unique = True

            for i, v in enumerate(metrics['per_layer_usage']):
                epoch_stats[f'Layer Usage/{i}'] = float(v)
            for i, v in enumerate(metrics['per_layer_entropy']):
                epoch_stats[f'Layer Entropy/{i}'] = float(v)

            model.train()

            if global_unique >= global_unique_threshold:
                logger.info(f'Early stopping at epoch {epoch}: global unique IDs >= {global_unique_threshold}')
                if config.general.use_wandb:
                    wandb.log(epoch_stats, step=epoch)
                results.append(epoch_stats)
                break

        if config.general.use_wandb and computed_global_unique:
            wandb.log(epoch_stats, step=epoch)

        epoch_progress.set_postfix(epoch_stats)
        results.append(epoch_stats)

    return results
