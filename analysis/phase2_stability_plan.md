# Phase 2 Plan: Stabilize Quantization Dynamics

## Purpose

Reduce codebook collapse and ensure later quantization layers contribute meaningful signal to uniqueness.

## What to change in this codebase

- Add residual normalization before each quantization step in [modules/rq_vae.py](modules/rq_vae.py).
- Introduce an optional layer-wise training schedule inside [train_rq_vae.py](train_rq_vae.py) (unlock layers progressively).
- Make loss scaling consistent so reconstruction and quantization terms are comparable across batch sizes.

## Where it fits

- Residual loop and quantization calls in [modules/rq_vae.py](modules/rq_vae.py).
- Training loop and loss aggregation in [train_rq_vae.py](train_rq_vae.py).
- Configuration defaults in the YAML files under [config](config).

## Implementation details

### 1) Residual normalization in the residual loop

In [modules/rq_vae.py](modules/rq_vae.py), inside the quantization loop:

```python
for layer in self.quantization_layers:
	if self.normalize_residuals:
		res = torch.nn.functional.normalize(res, dim=-1)
	quantized = layer(res, temperature=temperature)
	res = res - quantized.embeddings
```

Add a config flag normalize_residuals and pass it into the model constructor.

### 2) Layer-wise schedule in the trainer

In [train_rq_vae.py](train_rq_vae.py), gate layer updates:

```python
unlocked_layers = min(num_layers, 1 + epoch // steps_per_layer)
for idx, layer in enumerate(model.quantization_layers):
	layer.trainable = idx < unlocked_layers
```

In the quantization module, skip updates when not trainable.

### 3) Consistent loss scaling

In [modules/rq_vae.py](modules/rq_vae.py), ensure reductions are mean-based:

```python
recon_loss = torch.nn.functional.mse_loss(x_hat, x, reduction="mean")
commit_loss = quantized.quantize_loss.mean()
loss = recon_weight * recon_loss + commit_weight * commit_loss
```

Make recon_weight and commit_weight explicit config values.

## Required adaptations

- Add config flags for residual normalization and layer-wise schedule.
- Adjust loss reductions and ensure logging uses the same scale for all runs.
- Track per-layer usage and entropy (from Phase 1) while testing these changes.

## What to append

- A short ablation matrix for: baseline, residual normalization only, layer-wise only, both.
- A diagnostic log section to compare per-layer usage and uniqueness per epoch.

## Why this helps

- Residual normalization prevents later layers from seeing vanishing or unstable residuals, which often causes collapse.
- Layer-wise training gives early codebooks time to stabilize, improving downstream residual structure.
- Consistent loss scaling makes uniqueness changes attributable to model changes, not batch-size effects.

## Deliverables

- Configurable residual normalization in the RQ-VAE forward pass.
- Optional layer-wise training schedule in the trainer.
- Ablation report comparing uniqueness, coverage, and reconstruction.
