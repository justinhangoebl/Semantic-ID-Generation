import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb

def train(model, data, optimizer, scheduler, num_epochs, device, config):
    model.train()
    
    epoch_progress = tqdm(range(num_epochs), total=num_epochs, desc="Training Loop")
    results = []
    
    train_loader = DataLoader(data, batch_size=config.data.batch_size)
    
    for epoch in epoch_progress:
        total_loss = 0
        total_reconstruction_loss = 0
        total_commit_loss = 0
        p_unique = 0
        
        if(epoch == 0):
            kmeans_init_data = torch.Tensor(data[torch.arange(min(20000, len(data)))]).to(device, dtype=torch.float32)
            model(kmeans_init_data)
            
        for batch in train_loader:
            batch = batch.to(device).float()
            optimizer.zero_grad()
            result = model(batch)
            result.loss.backward()
            optimizer.step()
            #scheduler.step()
            
            total_loss += result.loss.item()
            total_reconstruction_loss += result.reconstruction_loss.item()
            total_commit_loss += result.rqvae_loss.item()
            p_unique += result.p_unique_ids.item()
            
        epoch_stats = {"Epoch": epoch,
                        "Loss": total_loss / len(train_loader),
                        "Reconstruction Loss": total_reconstruction_loss / len(train_loader),
                        "RQ-VAE Loss": total_commit_loss / len(train_loader),
                        "Prob Unique IDs": p_unique / len(train_loader)}
        if p_unique/ len(train_loader) >= 1:
            break
            
        
        if config.general.use_wandb:
            wandb.log(epoch_stats, step=epoch)
            
        epoch_progress.set_postfix(epoch_stats)
        
        results.append(epoch_stats)
    return results