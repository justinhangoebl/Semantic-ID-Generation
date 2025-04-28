# Stop on errors
$ErrorActionPreference = 'Stop'

# Create virtual environment
try {
    py -m venv .venv
    Write-Host "Virtual environment created." -ForegroundColor Green
}
catch {
    Write-Host "Failed to create virtual environment. Check if Python is installed." -ForegroundColor Red
    exit
}

# Activate the environment
try {
    .\.venv\Scripts\Activate.ps1
    Write-Host "Virtual environment activated." -ForegroundColor Green
}
catch {
    Write-Host "Failed to activate virtual environment. Check the path or execution policy." -ForegroundColor Red
    exit
}

python.exe -m pip install --upgrade pip

# Install packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install numpy pandas scikit-learn matplotlib ipykernel jupyterlab tqdm torch_geometric einops polars wandb sentence-transformers hydra-core datasets accelerate transformers dotenv

Write-Host "All dependencies installed successfully!" -ForegroundColor Green