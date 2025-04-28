import os
import wandb
from dotenv import load_dotenv
    
def wandb_login():
    load_dotenv()
    key = os.getenv("wandb_key")
    wandb.login(key=key)
    
if __name__ == "__main__":
    wandb_login()
    print("WandB login successful.")