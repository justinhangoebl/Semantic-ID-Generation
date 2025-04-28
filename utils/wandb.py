import os
import wandb
from dotenv import load_dotenv
from utils.model_id_generation import generate_model_id

def wandb_init(config):
    model_id = generate_model_id(config)
    wandb.init(
        project=config.general.wandb_project,
        entity=config.general.wandb_entity,
        config=config,
        id=model_id,
        resume="allow",
        name=model_id,
        dir=os.path.join("outputs", "wandb", "runs"),
    )

def wandb_login():
    load_dotenv()
    key = os.getenv("wandb_key")
    wandb.login(key=key)
    
if __name__ == "__main__":
    wandb_login()
    print("WandB login successful.")