from typing import Dict

def generate_model_id(config: Dict) -> str:
    dataset = config.data.dataset
    batch_size = config.data.batch_size
    normalize_data = config.data.normalize_data
    hidden_dimension = config.model.hidden_dimensions
    latent_dimension = config.model.latent_dimension
    num_codebook_layers = config.model.num_codebook_layers
    codebook_clusters = config.model.codebook_clusters
    commitment_weight = config.model.commitment_weight
    learning_rate = config.train.learning_rate
    weight_decay = config.train.weight_decay
    num_epochs = config.train.num_epochs
    
    # Create the model ID string
    model_id = (
        f"{dataset}-{config.data.category}-bs{batch_size}-norm{str(normalize_data)[0]}-"
        f"hd{'_'.join(map(str, hidden_dimension))}-ld{latent_dimension}-"
        f"cb{num_codebook_layers}x{codebook_clusters}-cw{commitment_weight}-"
        f"lr{learning_rate}-wd{weight_decay}-ep{num_epochs}"
    )
    
    return model_id