# Semantic-ID Generation using RQ-VAE

Semantic ID Generation based on RQ-VAE Architecture for Recommendation purposes based on the presented architecture in "Recommender Systems with Generative Retrieval" by Rajput S. et al. and "Better Generalization with Semantic IDs: A Case Study in Ranking for Recommendations" by Singh A. et al..

The items in the corpus are mapped to a tuple set of semantic IDs by training an RQ-VAE, see Figure below.

![proposed Architecture for RQ-VAE from Tiger](./assets/RQ-VAE-Architecture-Rajput-TIGER.jpg)

## Features

- **Multiple Quantization Methods**: Support for both Straight-Through Estimation (STE) and Gumbel Softmax quantization
- **Temperature Annealing**: Automatic temperature scheduling for Gumbel Softmax training
- **Joint Training Ready**: Optimized for joint training with language models for generative retrieval
- **Flexible Configuration**: YAML-based configuration system with extensive customization options
- **Comprehensive Testing**: Full test suite ensuring reliability and backward compatibility

## Quantization Methods

### Straight-Through Estimation (STE)

The traditional method using discrete quantization with gradient bypass:

- Fast and memory efficient
- Well-established in VQ-VAE literature
- May suffer from gradient mismatch issues

### Gumbel Softmax (Recommended for Joint Training)

A differentiable alternative that provides better gradient flow:

- Continuous relaxation of discrete sampling
- Better gradient flow for end-to-end training
- Temperature annealing for gradual transition from soft to hard assignments
- Ideal for joint training with language models

## Installation

1. Clone the repository

```bash
git clone <repository-url>
cd RQ-VAE
```

2. Run the `install.ps1` script to install all dependencies into a virtual environment:

```powershell
.\install.ps1
```

Or install manually:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install numpy pandas scikit-learn matplotlib ipykernel jupyterlab tqdm torch_geometric einops polars wandb sentence-transformers hydra-core datasets accelerate transformers dotenv pytest
```

## Quick Start

### Training with STE (Default)

```bash
python main.py --config config/config_ml100k.yaml
```

### Training with Gumbel Softmax

```bash
python main.py --config config/config_ml100k_gumbel.yaml
```

### Running Tests

```bash
python run_tests.py
```

## Configuration

The configuration system supports both quantization methods. Key parameters:

```yaml
model:
  quantization_method: "gumbel_softmax" # or "ste"
  temperature: 2.0 # Initial temperature for Gumbel Softmax
  min_temperature: 0.05 # Minimum temperature
  temperature_decay: 0.9995 # Decay rate per update

train:
  temperature_annealing: True # Enable temperature annealing
  temperature_update_frequency: 1 # Update every N epochs
```

## References

[Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065) by Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan H. Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Q. Tran, Jonah Samost, Maciej Kula, Ed H. Chi, Maheswaran Sathiamoorthy

[RQ-VAE Recommender - Inspirational Git Reposiory](https://github.com/EdoardoBotta/RQ-VAE-Recommender) by Edoardo Botta

[Better Generalization with Semantic IDs: A Case Study in Ranking for Recommendations](https://arxiv.org/abs/2306.08121) by Anima Singh, Trung Vu, Nikhil Mehta, Raghunandan Keshavan, Maheswaran Sathiamoorthy, Yilin Zheng, Lichan Hong, Lukasz Heldt, Li Wei, Devansh Tandon, Ed H. Chi, Xinyang Yi
