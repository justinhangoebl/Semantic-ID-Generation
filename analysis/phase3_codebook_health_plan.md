# Phase 3 Plan: Codebook Health and Diversity

## Purpose

Prevent dead codebook entries and improve coverage so semantic IDs remain diverse and interpretable.

## What to change in this codebase

- Add monitoring for dead codebook entries per layer (based on usage frequency).
- Add a lightweight reinitialization strategy for dead entries using recent residual samples.
- Add a soft diversity regularizer that discourages overly peaked code usage.

## Where it fits

- Codebook handling in [modules/quantization.py](modules/quantization.py).
- Residual tracking and usage logging in [modules/rq_vae.py](modules/rq_vae.py).
- Training loop hooks in [train_rq_vae.py](train_rq_vae.py).

## Implementation details

### 1) Track usage counts

In [modules/quantization.py](modules/quantization.py), add a buffer to count usage:

```python
self.register_buffer("usage_counts", torch.zeros(codebook_size))
```

Inside forward, after ids are computed:

```python
with torch.no_grad():
	self.usage_counts.scatter_add_(0, ids, torch.ones_like(ids, dtype=self.usage_counts.dtype))
```

### 2) Dead-code detection and reinit

In [train_rq_vae.py](train_rq_vae.py), on a schedule (e.g., every N epochs):

```python
dead_mask = layer.usage_counts < dead_threshold
dead_ids = dead_mask.nonzero(as_tuple=True)[0]
if dead_ids.numel() > 0:
	sampled = residuals[torch.randint(0, residuals.size(0), (dead_ids.numel(),))]
	layer.embedding.weight.data[dead_ids] = sampled
	layer.usage_counts[dead_ids] = 0
```

Add a cooldown period so reinit does not repeat too often.

### 3) Diversity regularizer

Compute a usage-based entropy bonus in [train_rq_vae.py](train_rq_vae.py):

```python
counts = layer.usage_counts.float()
probs = counts / counts.sum().clamp_min(1.0)
entropy = -(probs * (probs + 1e-9).log()).sum()
loss = loss - entropy_weight * entropy
```

## Required adaptations

- Track per-layer code usage counts during training (batch-aggregated).
- Store rolling windows of residuals to reinitialize dead codes safely.
- Add configuration for thresholds and reinit frequency.

## What to append

- A dead-code report per epoch (count and percentage per layer).
- A small summary that correlates codebook usage with uniqueness changes.

## Why this helps

- Dead codebook entries reduce the effective capacity of the semantic ID space.
- Reinitialization recovers capacity without changing the global architecture.
- Diversity regularization increases coverage and usually improves global uniqueness.

## Deliverables

- Dead-code monitoring and reinitialization logic in quantization modules.
- Configurable diversity regularization.
- Logs showing usage distribution before and after changes.
